// ABOUTME: --env / --env-file flag registration and parsing for the lifecycle
// ABOUTME: verbs: KEY=VAL lines read from a file or stdin, so a secret's value
// ABOUTME: never lands on yoloai's argv, which other local accounts can read.
package lifecycle

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"

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

// containsUnsplitLineTerminator reports whether line holds a line terminator
// this parser does not split on. Unicode's line-termination set is LF, VT, FF,
// CR, CRLF, NEL, LS and PS (UAX #13 §4.1); parseEnvFileData splits on LF and
// strips a trailing CR, so everything else in that set is a line ending it
// cannot see — and a file written with one is a single line here, which is the
// outcome the CR rule exists to refuse.
//
// Written as codepoints rather than a string literal: three of these are
// invisible, and a literal would be bytes in the source for the next reader to
// take on trust.
//
// The lone-0x85 case is why this decodes by hand. NEL is 0x85 in Latin-1,
// CP1252 and anything converted from EBCDIC, where it is not valid UTF-8 — so a
// rune-wise scan sees RuneError and the U+0085 test never fires. Checked as a
// byte, but only for a byte that is already invalid UTF-8, so a Latin-1 value
// byte that is not NEL (an é, 0xE9) is left alone: this parser deliberately does
// not insist a value is valid UTF-8, because a password need not be.
func containsUnsplitLineTerminator(line string) bool {
	for i := 0; i < len(line); {
		r, size := utf8.DecodeRuneInString(line[i:])
		if r == utf8.RuneError && size == 1 {
			if line[i] == nelByte {
				return true
			}
			i++
			continue
		}
		if isUnsplitLineTerminator(r) {
			return true
		}
		i += size
	}
	return false
}

// nelByte is NEL as a single byte, which is how a legacy encoding carries it.
const nelByte = 0x85

// isUnsplitLineTerminator: VT, FF, NEL, LS, PS. Not LF (the split) and not CR
// (its own check and its own message, because CRLF is legitimate and a lone CR
// is the common mistake worth naming).
func isUnsplitLineTerminator(r rune) bool {
	return r == '\v' || r == '\f' || r == 0x0085 || r == 0x2028 || r == 0x2029
}

// createEnvUsage is the --env/--env-file help for the create verbs (new, run).
var createEnvUsage = envFlagUsage{
	env:     "Environment variable (KEY=VAL, repeatable). Not for secrets — the value is on the command line of every invocation you pass it to, which another local user can usually read with 'ps'. Use --env-file",
	envFile: "Read environment variables from a file of KEY=VAL lines, or from stdin with '-'. The way to pass a secret: the value never reaches the command line",
}

// perStartEnvUsage builds the --env/--env-file help for start and restart, whose
// values apply only to the launch they are passed to. verb is "start" or
// "restart", so each verb's help names the command the user has to repeat.
func perStartEnvUsage(verb string) envFlagUsage {
	return envFlagUsage{
		env:     "Per-sandbox env var KEY=VAL (not persisted; re-supply on each " + verb + "). Not for secrets — the value is usually visible to other local users via 'ps' while this command runs. Use --env-file",
		envFile: "Read per-sandbox env vars from a file of KEY=VAL lines, or from stdin with '-' (not persisted; re-supply on each " + verb + "). The way to pass a secret",
	}
}

// resetEnvUsage is the --env/--env-file help for reset, where the values only
// take effect on the --restart path.
var resetEnvUsage = envFlagUsage{
	env:     "Per-sandbox env var KEY=VAL applied on --restart (not persisted). Not for secrets — the value is usually visible to other local users via 'ps' while this command runs. Use --env-file",
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
//
// Reading Changed is the only such read in the CLI — every other string flag
// treats "" as absent via cliutil.FlagStr. If a second secret-bearing flag needs
// the same distinction, that asymmetry is worth a shared helper rather than a
// second copy of this.
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
	// Only expand where there is something to expand. ExpandPath is a no-op for a
	// path with neither a leading ~ nor a ${, and reaching cliutil.Layout() for
	// one would make an ordinary path depend on process-wide state.
	if !strings.HasPrefix(path, "~") && !strings.Contains(path, "${") {
		return path, nil
	}
	layout := cliutil.Layout()
	expanded, err := config.ExpandPath(path, layout.HomeDir, layout.Env().EnvForConfigInterpolation())
	if err != nil {
		// %s, not %w: standards/go.md asks upstream errors to be wrapped, and no
		// NewUsageError call site in this repo does it (0 of them), so wrapping
		// only here would make this the single place errors.Is reaches through a
		// UsageError. The declared rule and the practised baseline disagree;
		// resolving that is a decision about yoerrors, not about this flag, so the
		// divergence is stated rather than settled (development-principles.md §1).
		// Nothing is lost that a reader needs: ExpandPath's own message carries
		// the cause in full — the variable it could not resolve, or the ${ it
		// could not find a '}' for.
		return "", yoerrors.NewUsageError("invalid --env-file path: %s", err)
	}
	// A ${VAR} on the interpolation allowlist can be set and empty, and an empty
	// path here would land back on the silent "no file" this function refuses.
	if expanded == "" {
		return "", yoerrors.NewUsageError("--env-file %s expanded to an empty path", path)
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
//     anywhere else is an error, checked before blank and comment lines are
//     skipped: a CR-only (classic Mac) file is one "line" to this parser, so
//     guessing would turn the whole file into one variable whose value is the rest
//     of the secrets — or, if that line starts with '#', into nothing at all.
//   - Any other line terminator the LF split cannot see — VT, FF, NEL, LS, PS,
//     including NEL as the lone 0x85 byte a legacy encoding carries — is an
//     error, in the same place and on the same argument as the CR rule
//     (containsUnsplitLineTerminator). Lines are split on LF, so this list is
//     Unicode's line-termination set minus LF and the CRLF pair, and the rule is
//     the set rather than whichever codepoint someone reported.
//   - A leading UTF-8 BOM is dropped. Editors write it invisibly, and it would
//     otherwise make the first line's key unmatchable for a reason the user
//     cannot see.
//   - A value need not be valid UTF-8. A password in a legacy encoding is still a
//     password, so only the bytes above are refused, not mojibake in general.
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
		line := strings.TrimSuffix(raw, "\r")

		// Before anything is allowed to ignore the line. A CR-only file is a
		// single line to a parser that splits on LF, and if it begins with '#' —
		// the ordinary first line of a secrets file — skipping comments first
		// would discard the entire file and start the sandbox with an empty
		// environment, silently and with exit 0. That is the outcome this parser
		// exists to prevent, so the check cannot sit behind the skip.
		if strings.ContainsRune(line, '\r') {
			return nil, yoerrors.NewUsageError("--env-file line %d: contains a carriage return — CRLF line endings are fine, a CR-only file is not", lineNo)
		}
		// And the same rule for every other line terminator the LF split cannot
		// see — VT, FF, NEL, LS, PS (containsUnsplitLineTerminator). Each makes a
		// file a single line here: `A=1<LS>B=2` became one variable holding the
		// rest of the secrets, and a '#' first line made the whole file vanish
		// into the comment skip. That is the CR defect exactly, one encoding over,
		// so it gets the CR rule rather than a narrower one aimed at whichever
		// codepoint was reported. Named by codepoint because there is nothing to
		// show.
		if containsUnsplitLineTerminator(line) {
			return nil, yoerrors.NewUsageError("--env-file line %d: contains a line terminator this parser does not split on (VT, FF, U+0085, U+2028 or U+2029) — use LF or CRLF line endings", lineNo)
		}
		if strings.ContainsRune(line, 0) {
			return nil, yoerrors.NewUsageError("--env-file line %d: contains a NUL byte, which no environment variable can carry", lineNo)
		}

		// TrimLeft only: indentation is not content, but trailing whitespace is
		// part of the value.
		line = strings.TrimLeft(line, " \t")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
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
