// ABOUTME: --env / --env-file flag registration and parsing for the lifecycle
// ABOUTME: verbs: KEY=VAL lines read from a file or stdin, so a secret's value
// ABOUTME: never lands on yoloai's argv where any other local account reads it.
package lifecycle

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/kstenerud/yoloai/internal/cli/cliutil"
	"github.com/kstenerud/yoloai/internal/config"
	"github.com/kstenerud/yoloai/yoerrors"
	"github.com/spf13/cobra"
)

// envFlagUsage holds one verb's help lines for the --env / --env-file pair.
// The two flags are registered together (addEnvFlags) so a verb cannot acquire
// one without the other, and so their types cannot drift apart again: --env was
// once a StringSlice on `new` and a StringArray on the other three, which split
// every value on commas for one verb only (DF195).
type envFlagUsage struct {
	env     string
	envFile string
}

// envVarNamePattern is the POSIX-shaped name a --env-file key must match. It is
// in the error message as well, so it lives in one place.
const envVarNamePattern = "[A-Za-z_][A-Za-z0-9_]*"

// stdinPath is the conventional "read it from stdin" flag value, already used by
// --prompt and --prompt-file.
const stdinPath = "-"

var envVarNameRe = regexp.MustCompile("^" + envVarNamePattern + "$")

// createEnvUsage is the --env/--env-file help for the create verbs (new, run).
var createEnvUsage = envFlagUsage{
	env:     "Environment variable (KEY=VAL, repeatable). Not for secrets — the value is on the command line of every invocation you pass it to, where any local user can read it with 'ps'. Use --env-file",
	envFile: "Read environment variables from a file of KEY=VAL lines, or from stdin with '-'. The way to pass a secret: the value never reaches the command line",
}

// perStartEnvUsage builds the --env/--env-file help for start and restart, whose
// values apply only to the launch they are passed to. verb is "start" or
// "restart", so each verb's help names the command the user has to repeat.
func perStartEnvUsage(verb string) envFlagUsage {
	return envFlagUsage{
		env:     "Per-sandbox env var KEY=VAL (not persisted; re-supply on each " + verb + "). Not for secrets — the value is visible to any local user via 'ps' while this command runs. Use --env-file",
		envFile: "Read per-sandbox env vars from a file of KEY=VAL lines, or from stdin with '-' (not persisted; re-supply on each " + verb + "). The way to pass a secret",
	}
}

// resetEnvUsage is the --env/--env-file help for reset, where the values only
// take effect on the --restart path.
var resetEnvUsage = envFlagUsage{
	env:     "Per-sandbox env var KEY=VAL applied on --restart (not persisted). Not for secrets — the value is visible to any local user via 'ps' while this command runs. Use --env-file",
	envFile: "Read per-sandbox env vars applied on --restart from a file of KEY=VAL lines, or from stdin with '-' (not persisted). The way to pass a secret",
}

// addEnvFlags registers the --env / --env-file pair on cmd.
func addEnvFlags(cmd *cobra.Command, u envFlagUsage) {
	// StringArray, never StringSlice: one occurrence is one variable, and a
	// comma is ordinary inside a value (NO_PROXY=localhost,127.0.0.1) — DF195.
	cmd.Flags().StringArray("env", nil, u.env)
	cmd.Flags().String("env-file", "", u.envFile)
}

// resolveEnvFromFlags reads --env and --env-file off cmd and returns the merged
// environment. Single entry point for every verb that takes the pair, so the
// precedence rule (there is none — a key in both is an error) and the
// stdin-contention check cannot differ between them.
func resolveEnvFromFlags(cmd *cobra.Command) (map[string]string, error) {
	envSlice, _ := cmd.Flags().GetStringArray("env")
	envFile, err := envFilePath(cmd)
	if err != nil {
		return nil, err
	}
	if err := checkEnvFileStdinContention(cmd, envFile); err != nil {
		return nil, err
	}
	return resolveEnv(envSlice, envFile, cmd.InOrStdin())
}

// envFilePath returns the --env-file value, "" when the flag was not given, and
// the path with ~ and ${VAR} expanded as --prompt-file expands them.
//
// "Not given" is read from Changed, not from the value being empty, because an
// empty value is something a user can produce: `--env-file "$SECRETS"` with
// SECRETS unset. Treating that as "no file" starts a sandbox carrying none of the
// secrets it was told to carry, says nothing, and exits 0. --prompt-file does
// read "" as absent; for secret delivery, silence is the wrong direction.
func envFilePath(cmd *cobra.Command) (string, error) {
	f := cmd.Flags().Lookup("env-file")
	if f == nil || !f.Changed {
		return "", nil
	}
	path := f.Value.String()
	switch path {
	case "":
		return "", yoerrors.NewUsageError("--env-file needs a path, or - for stdin")
	case stdinPath:
		return path, nil
	}
	layout := cliutil.Layout()
	expanded, err := config.ExpandPath(path, layout.HomeDir, layout.Env().EnvForConfigInterpolation())
	if err != nil {
		return "", yoerrors.NewUsageError("invalid --env-file path: %s", err)
	}
	return expanded, nil
}

// checkEnvFileStdinContention refuses `--env-file -` alongside a prompt flag
// that also reads stdin. Both would consume the same stream, and the one that
// read second would silently see nothing — so this is an error, not a race.
//
// Hand-rolled rather than cobra's MarkFlagsMutuallyExclusive, which cannot
// express it: the flags are compatible, and only the one *value* they share is
// not. FlagStr yields "" for a flag the verb does not have (reset has no prompt).
func checkEnvFileStdinContention(cmd *cobra.Command, envFile string) error {
	if envFile != stdinPath {
		return nil
	}
	for _, name := range []string{"prompt", "prompt-file"} {
		if cliutil.FlagStr(cmd, name) == stdinPath {
			return yoerrors.NewUsageError("--env-file - and --%s - both read stdin: give one of them a file", name)
		}
	}
	return nil
}

// resolveEnv merges the --env values with the --env-file (or stdin) ones.
//
// A key supplied through both is an error. The alternative is a precedence rule,
// which is silent by nature: the user who passes the same key twice has two
// values in mind and gets one, with nothing to indicate which.
func resolveEnv(envSlice []string, envFile string, stdin io.Reader) (map[string]string, error) {
	envMap, err := parseEnvSlice(envSlice)
	if err != nil {
		return nil, err
	}
	if envFile == "" {
		return envMap, nil
	}

	data, err := readEnvFileData(envFile, stdin)
	if err != nil {
		return nil, err
	}
	fileMap, err := parseEnvFileData(data)
	if err != nil {
		return nil, err
	}

	var both []string
	for k := range fileMap {
		if _, clash := envMap[k]; clash {
			both = append(both, k)
		}
	}
	if len(both) > 0 {
		// Naming the keys is safe and is the only actionable part: each one was
		// typed as a --env argument, so it is already on the command line this
		// error is about. Their values are not named, here or anywhere else.
		slices.Sort(both)
		return nil, yoerrors.NewUsageError("%s set by both --env and --env-file: remove the duplicate, neither wins", strings.Join(both, ", "))
	}
	for k, v := range fileMap {
		envMap[k] = v
	}
	return envMap, nil
}

// readEnvFileData reads the raw --env-file bytes, from stdin when the path is
// "-". The path itself appears in the error: it came from the command line, so
// it is not what this flag exists to keep off it.
//
// Both read failures are operational, so they are plain wrapped errors (exit 1)
// rather than a UsageError: the argument was well formed and the file is not a
// config file (standards/go.md). The sibling --prompt-file does the same.
func readEnvFileData(path string, stdin io.Reader) ([]byte, error) {
	if path == stdinPath {
		data, err := io.ReadAll(stdin)
		if err != nil {
			return nil, fmt.Errorf("read --env-file from stdin: %w", err)
		}
		return data, nil
	}
	data, err := os.ReadFile(path) //nolint:gosec // G304: the path is the user-supplied --env-file value
	if err != nil {
		return nil, fmt.Errorf("read --env-file: %w", err)
	}
	// File permissions are the caller's business, deliberately: this reads a
	// path the user chose, and refusing a group-readable file would only move
	// the secret back onto the command line, which is strictly worse.
	return data, nil
}

// parseEnvFileData parses KEY=VAL lines.
//
// The rules, in full:
//
//   - A blank line, or one whose first non-whitespace character is '#', is
//     ignored. A '#' anywhere else is part of the value — passwords contain
//     them, and stripping an inline comment would silently truncate one.
//   - The value is literal to end of line: no quote stripping, no $VAR or
//     dotenv-style expansion, trailing whitespace kept. What you wrote is what
//     the sandbox gets.
//   - A trailing CR is dropped, so a CRLF file parses as the same thing a LF
//     file does rather than appending an invisible byte to every value. A CR
//     anywhere else is an error: a CR-only (classic Mac) file is one "line" to
//     this parser, and guessing would turn the whole file into one variable whose
//     value is the rest of the secrets.
//   - A leading UTF-8 BOM is dropped. Editors write it invisibly, and it would
//     otherwise make the first line's key unmatchable for a reason the user
//     cannot see.
//   - A NUL is an error: no environment can carry one, so accepting it only moves
//     the failure to exec, far from the file that caused it.
//   - The key must be a plain variable name (envVarNamePattern). `export FOO=1`,
//     a quoted name and `FOO = 1` are all rejected rather than passed through as
//     a variable nothing can read.
//   - A key set twice in one file is an error, for the same reason a key in both
//     --env and --env-file is: there is no silent last-wins.
//
// No error names anything the file contained — not a line, not even a key. The
// text goes to stderr and into a bug report's exit line, and a secret pasted into
// the key position is exactly the kind of accident that would ride along. Line
// numbers locate the problem without copying it, and where the cause is a byte
// the user cannot see, the message says so instead of showing it.
func parseEnvFileData(data []byte) (map[string]string, error) {
	out := make(map[string]string)
	setAt := make(map[string]int)
	text := strings.TrimPrefix(string(data), "\ufeff")
	for i, raw := range strings.Split(text, "\n") {
		lineNo := i + 1
		// TrimLeft only: indentation is not content, but trailing whitespace is
		// part of the value.
		line := strings.TrimLeft(strings.TrimSuffix(raw, "\r"), " \t")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.ContainsRune(line, '\r') {
			return nil, yoerrors.NewUsageError("--env-file line %d: contains a carriage return — CRLF line endings are fine, a CR-only file is not", lineNo)
		}
		if strings.ContainsRune(line, 0) {
			return nil, yoerrors.NewUsageError("--env-file line %d: contains a NUL byte, which no environment variable can carry", lineNo)
		}

		key, val, ok := strings.Cut(line, "=")
		if !ok {
			return nil, yoerrors.NewUsageError("--env-file line %d: expected KEY=VAL", lineNo)
		}
		if !envVarNameRe.MatchString(key) {
			return nil, yoerrors.NewUsageError("--env-file line %d: not a variable name (expected %s), so nothing before the '=' can be set — check for a space before the '=', an 'export ' prefix, quotes, or an invisible character such as a non-breaking space", lineNo, envVarNamePattern)
		}
		if prev, dup := setAt[key]; dup {
			return nil, yoerrors.NewUsageError("--env-file line %d: sets the same variable as line %d", lineNo, prev)
		}
		setAt[key] = lineNo
		out[key] = val
	}
	return out, nil
}
