// ABOUTME: Tests for --env-file: the parse rules, the stdin form, the refusal of
// ABOUTME: a key supplied twice, and the rule that nothing the file contained is
// ABOUTME: ever echoed — the property that keeps a secret out of logs and reports.
package lifecycle

import (
	"bytes"
	"errors"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/kstenerud/yoloai/internal/cli/clitest"
	"github.com/kstenerud/yoloai/yoerrors"
)

// envFileSecret is the value every test passes through --env-file. It carries a
// '#' so a parser that strips inline comments mangles it, and it is the string
// asserted absent from error messages and log output.
const envFileSecret = "s3cret#value "

// The line terminators this parser refuses because it does not split on them.
// Named rather than pasted: an invisible byte in a fixture is a fixture the next
// reader has to take on trust, and these three are indistinguishable by eye.
var (
	lsSep  = string(rune(0x2028)) // LINE SEPARATOR
	psSep  = string(rune(0x2029)) // PARAGRAPH SEPARATOR
	nelSep = string(rune(0x0085)) // NEXT LINE, UTF-8 (0xC2 0x85)
	vtSep  = "\v"                 // LINE TABULATION
	ffSep  = "\f"                 // FORM FEED
	// NEL as a single byte, which is how Latin-1, CP1252 and an
	// EBCDIC conversion carry it. Invalid UTF-8, so a rune-wise scan sees
	// RuneError and never the codepoint — the case that needs its own fixture.
	nelByteSep = "\x85"
)

// writeEnvFile writes body to a temp file and returns its path.
func writeEnvFile(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "env")
	require.NoError(t, os.WriteFile(path, []byte(body), 0o600))
	return path
}

func TestParseEnvFileData_Rules(t *testing.T) {
	tests := []struct {
		name string
		body string
		want map[string]string
	}{
		{"blank lines and comments", "# a comment\n\nA=1\n", map[string]string{"A": "1"}},
		{"indented comment", "   \t# a comment\n  A=1\n", map[string]string{"A": "1"}},
		{"whitespace-only line", "A=1\n   \t\n", map[string]string{"A": "1"}},
		{"a # inside a value is value", "A=pa#ss\n", map[string]string{"A": "pa#ss"}},
		{"quotes are literal", "A=\"quoted\"\nB='single'\n", map[string]string{"A": `"quoted"`, "B": "'single'"}},
		{"no $VAR expansion", "A=$HOME\n", map[string]string{"A": "$HOME"}},
		{"trailing whitespace is value", "A=x  \n", map[string]string{"A": "x  "}},
		{"= inside a value", "A=a=b\n", map[string]string{"A": "a=b"}},
		{"empty value", "A=\n", map[string]string{"A": ""}},
		{"CRLF line endings", "A=1\r\nB=2\r\n", map[string]string{"A": "1", "B": "2"}},
		{"no trailing newline", "A=1", map[string]string{"A": "1"}},
		{"empty file", "", map[string]string{}},
		{"comments only", "#one\n#two\n", map[string]string{}},
		// Editors write a BOM invisibly. Without stripping it the first key is
		// unmatchable for a reason the user cannot see in their own file.
		{"leading UTF-8 BOM", "\ufeffA=1\nB=2\n", map[string]string{"A": "1", "B": "2"}},
		// A value need not be valid UTF-8 — a password in a legacy encoding is
		// still a password. Only the byte that is a *line terminator* is refused,
		// so this must keep parsing or the NEL-byte check has over-reached.
		{"a non-NEL invalid byte is value", "A=caf\xe9\n", map[string]string{"A": "caf\xe9"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := parseEnvFileData([]byte(tc.body))
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

// TestParseEnvFileData_Errors covers every rejection, and holds each error to the
// rule that matters more than its wording: the message locates the bad line
// without quoting it. The error reaches stderr and a bug report's exit line,
// which has no redactor of its own, and the text it would be quoting is a secret.
func TestParseEnvFileData_Errors(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{"no equals sign", envFileSecret + "\n", "line 1: expected KEY=VAL"},
		{"export prefix", "export A=" + envFileSecret + "\n", "line 1: not a variable name"},
		{"space before the equals", "A = " + envFileSecret + "\n", "line 1: not a variable name"},
		{"quoted name", `"A"=` + envFileSecret + "\n", "line 1: not a variable name"},
		{"name starting with a digit", "1A=" + envFileSecret + "\n", "line 1: not a variable name"},
		{"name with a dash", "A-B=" + envFileSecret + "\n", "line 1: not a variable name"},
		{"same key twice", "A=1\n# a comment\nA=" + envFileSecret + "\n", "line 3: sets the same variable as line 1"},
		{"bad line after good ones", "A=1\nB=2\n" + envFileSecret + "\n", "line 3: expected KEY=VAL"},
		// A CR-only (classic Mac) file is one line to a LF-splitting parser.
		// Guessing would make the whole file one variable whose value is the rest
		// of the secrets — silently, with the other keys simply missing.
		{"CR-only line endings", "A=1\rB=" + envFileSecret + "\r", "line 1: contains a carriage return"},
		{"CR inside a value", "A=x\ry=" + envFileSecret + "\n", "line 1: contains a carriage return"},
		// The CR checks have to run before comments are skipped. A '#' header is
		// the ordinary first line of a secrets file, and a CR-only file is one
		// line: skipping first discarded the whole file as a comment and started
		// the sandbox with an empty environment, silently, exit 0.
		{"CR-only file starting with a comment", "# secrets\rA=" + envFileSecret + "\r", "line 1: contains a carriage return"},
		{"CR-only file starting with an indented comment", "  # secrets\rA=" + envFileSecret + "\r", "line 1: contains a carriage return"},
		// The CR rule's own argument applies to every line terminator this parser
		// does not split on, and stopping at CR left the same defect one encoding
		// over: `A=1<LS>B=2` was one variable holding the rest of the secrets, and
		// a '#' first line sent the whole file into the comment skip — an empty
		// environment, silently, exit 0.
		{"LS-separated file", "A=1" + lsSep + "B=" + envFileSecret + "\n", "line 1: contains a line terminator"},
		{"PS-separated file", "A=1" + psSep + "B=" + envFileSecret + "\n", "line 1: contains a line terminator"},
		{"NEL-separated file", "A=1" + nelSep + "B=" + envFileSecret + "\n", "line 1: contains a line terminator"},
		{"VT-separated file", "A=1" + vtSep + "B=" + envFileSecret + "\n", "line 1: contains a line terminator"},
		{"FF-separated file", "A=1" + ffSep + "B=" + envFileSecret + "\n", "line 1: contains a line terminator"},
		// Invalid UTF-8, so the codepoint check cannot see it.
		{"NEL as a lone legacy byte", "A=1" + nelByteSep + "B=" + envFileSecret + "\n", "line 1: contains a line terminator"},
		{"LS-separated file starting with a comment", "# secrets" + lsSep + "A=" + envFileSecret + lsSep, "line 1: contains a line terminator"},
		{"FF-separated file starting with a comment", "# secrets" + ffSep + "A=" + envFileSecret + "\n", "line 1: contains a line terminator"},
		{"NUL byte", "A=x\x00" + envFileSecret + "\n", "line 1: contains a NUL byte"},
		{"NUL byte in a comment", "#x\x00" + envFileSecret + "\nA=1\n", "line 1: contains a NUL byte"},
		// An invisible character cannot be shown in a message that must not quote
		// the line, so the message names the possibility instead.
		{"non-breaking space before the key", "\u00a0A=" + envFileSecret + "\n", "invisible character"},
		{"tab inside the key", "A\tB=" + envFileSecret + "\n", "not a variable name"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := parseEnvFileData([]byte(tc.body))
			assertUsageError(t, err, tc.wantErr)
			assert.NotContains(t, err.Error(), strings.TrimSpace(envFileSecret),
				"the error quotes the file's own text, which is the secret")
		})
	}
}

func TestResolveEnv_ReadsFileAndStdin(t *testing.T) {
	body := "# the secret\nAPI_TOKEN=" + envFileSecret + "\nOTHER=plain\n"
	want := map[string]string{"API_TOKEN": envFileSecret, "OTHER": "plain"}

	t.Run("from a file", func(t *testing.T) {
		t.Parallel()
		got, err := resolveEnv(nil, writeEnvFile(t, body), nil)
		require.NoError(t, err)
		assert.Equal(t, want, got)
	})

	t.Run("from stdin", func(t *testing.T) {
		t.Parallel()
		got, err := resolveEnv(nil, "-", strings.NewReader(body))
		require.NoError(t, err)
		assert.Equal(t, want, got)
	})

	t.Run("merged with --env on other keys", func(t *testing.T) {
		t.Parallel()
		got, err := resolveEnv([]string{"PLAIN=1"}, "-", strings.NewReader(body))
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"API_TOKEN": envFileSecret, "OTHER": "plain", "PLAIN": "1"}, got)
	})

	// "" is resolveEnv's internal encoding of "the flag was not given"; what the
	// user typed is decided by envFilePath, which refuses an empty --env-file
	// rather than reaching here (TestEnvFilePath).
	t.Run("absent env-file leaves --env alone", func(t *testing.T) {
		t.Parallel()
		got, err := resolveEnv([]string{"PLAIN=1"}, "", nil)
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"PLAIN": "1"}, got)
	})

	// A failed read is operational, not a usage error: the flag was well formed
	// and the file is not a config file (standards/go.md), and --prompt-file's
	// equivalent is a plain wrapped error too. The category is the assertion —
	// it decides the process's exit code.
	t.Run("unreadable file is a plain wrapped error, not a usage error", func(t *testing.T) {
		t.Parallel()
		missing := filepath.Join(t.TempDir(), "absent")
		_, err := resolveEnv(nil, missing, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "read --env-file")
		var ue *yoerrors.UsageError
		assert.False(t, errors.As(err, &ue), "a missing file is an I/O failure, not a usage error")
		assert.ErrorIs(t, err, fs.ErrNotExist, "the cause has to survive: wrapped with %w, not flattened with %s")
	})
}

// TestEnvFilePath covers what the flag's value means before anything reads it.
func TestEnvFilePath(t *testing.T) {
	t.Run("not given is no file", func(t *testing.T) {
		t.Parallel()
		got, err := envFilePath(NewStartCmd())
		require.NoError(t, err)
		assert.Empty(t, got)
	})

	// The way to produce this is `--env-file "$SECRETS"` with SECRETS unset. Read
	// as "no file" it starts a sandbox carrying none of the secrets it was told to
	// carry and exits 0 — the one outcome this flag exists to prevent.
	t.Run("explicitly empty is refused, not read as absent", func(t *testing.T) {
		t.Parallel()
		cmd := NewStartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", ""}))
		_, err := envFilePath(cmd)
		assertUsageError(t, err, "--env-file needs a path")
	})

	t.Run("stdin passes through unexpanded", func(t *testing.T) {
		t.Parallel()
		cmd := NewStartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", "-"}))
		got, err := envFilePath(cmd)
		require.NoError(t, err)
		assert.Equal(t, "-", got)
	})

	// Matches --prompt-file, which expands via config.ExpandPath. A quoted
	// ~/secrets.env reaches the flag unexpanded by the shell.
	//
	// Not parallel: clitest.Home sets a process-wide Layout.
	t.Run("tilde is expanded as on --prompt-file", func(t *testing.T) {
		home := clitest.Home(t)
		cmd := NewStartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", "~/secrets.env"}))
		got, err := envFilePath(cmd)
		require.NoError(t, err)
		assert.Equal(t, filepath.Join(home, "secrets.env"), got)
	})

	// The same silence, one hop later: a ${VAR} on the interpolation allowlist can
	// be set and empty, and an empty expansion would be read as "no file" by the
	// caller. Refusing only the pre-expansion value would watch a proxy.
	t.Run("a ${VAR} that expands to nothing is refused too", func(t *testing.T) {
		t.Setenv("TZ", "")
		clitest.Home(t)
		cmd := NewStartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", "${TZ}"}))
		_, err := envFilePath(cmd)
		assertUsageError(t, err, "expanded to an empty path")
	})

	// An ordinary path must not reach cliutil.Layout() at all: it panics when the
	// root Layout is unset, which would make every caller of this depend on
	// process-wide state for a path that needs no expansion.
	t.Run("an ordinary path needs no Layout", func(t *testing.T) {
		cmd := NewStartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", "/tmp/secrets.env"}))
		got, err := envFilePath(cmd)
		require.NoError(t, err)
		assert.Equal(t, "/tmp/secrets.env", got)
	})
}

// TestResolveEnv_SameKeyInBothIsAnError pins the refusal rather than a precedence
// rule: a user who supplies one key twice has two values in mind, and silently
// keeping one of them is the failure mode the refusal exists to prevent.
func TestResolveEnv_SameKeyInBothIsAnError(t *testing.T) {
	t.Run("one key", func(t *testing.T) {
		t.Parallel()
		_, err := resolveEnv([]string{"API_TOKEN=from-flag"}, "-", strings.NewReader("API_TOKEN="+envFileSecret+"\n"))
		assertUsageError(t, err, "API_TOKEN set by both --env and --env-file")
		assert.NotContains(t, err.Error(), strings.TrimSpace(envFileSecret))
		assert.NotContains(t, err.Error(), "from-flag")
	})

	t.Run("several keys, named in order", func(t *testing.T) {
		t.Parallel()
		_, err := resolveEnv(
			[]string{"B=1", "A=2", "KEEP=3"},
			"-", strings.NewReader("A=x\nB=y\nUNIQUE=z\n"))
		assertUsageError(t, err, "A, B set by both --env and --env-file")
		assert.NotContains(t, err.Error(), "KEEP")
		assert.NotContains(t, err.Error(), "UNIQUE")
	})
}

// envVerbs is every verb that takes an environment: the four --env-file had to
// reach, plus run, which shares new's creation flags and the same exposed argv.
var envVerbs = []string{"new", "run", "start", "restart", "reset"}

// newEnvVerbCmd builds one verb's command. A fresh one per case: cobra's
// StringArray accumulates across Parse calls, so a reused command carries the
// previous case's --env.
func newEnvVerbCmd(t *testing.T, verb string) *cobra.Command {
	t.Helper()
	switch verb {
	case "new":
		return NewNewCmd("test")
	case "run":
		return NewRunCmd("test")
	case "start":
		return NewStartCmd()
	case "restart":
		return NewRestartCmd()
	case "reset":
		return NewResetCmd()
	}
	t.Fatalf("unknown verb %q", verb)
	return nil
}

// TestEnvFileFlag_ReachesEveryVerb is the ticket's own claim, one verb at a time:
// the value arrives in the environment the sandbox is given, and no flag on the
// command holds it — only the path.
//
// The assert.Equal is the measuring assertion. The flag-set walk below it is a
// **tripwire, not evidence**: this test only ever puts a path on the command line,
// so no flag value can hold the secret whatever the resolver does, and gutting the
// resolver leaves the walk green. It is here to fail a future change that gives
// --env-file an inline form, and it replaced an assertion over a literal slice the
// test itself wrote, which was the same shape with none of that forward value.
func TestEnvFileFlag_ReachesEveryVerb(t *testing.T) {
	for _, verb := range envVerbs {
		t.Run(verb, func(t *testing.T) {
			t.Parallel()
			cmd := newEnvVerbCmd(t, verb)
			path := writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")
			require.NoError(t, cmd.Flags().Parse([]string{"--env-file", path}))

			got, err := resolveEnvFromFlags(cmd)
			require.NoError(t, err)
			assert.Equal(t, map[string]string{"API_TOKEN": envFileSecret}, got)

			secret := strings.TrimSpace(envFileSecret)
			cmd.Flags().VisitAll(func(f *pflag.Flag) {
				assert.NotContains(t, f.Value.String(), secret,
					"--%s carries the value; only the path may be on the command line", f.Name)
			})
		})
	}
}

// TestResolveOptions_EnvReachesEachVerbsLibraryCall covers the per-verb wiring the
// shared resolver cannot: each verb builds its own options struct, and a verb that
// resolved the environment and then failed to put it in that struct would deliver
// an empty environment with nothing failing. One test per call site, which is what
// makes a revert of any single one of them go red.
func TestResolveOptions_EnvReachesEachVerbsLibraryCall(t *testing.T) {
	want := map[string]string{"API_TOKEN": envFileSecret}

	t.Run("start", func(t *testing.T) {
		t.Parallel()
		cmd := NewStartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")}))
		got, err := resolveStartOptions(cmd, &startOpts{})
		require.NoError(t, err)
		assert.Equal(t, want, got.Env)
	})

	t.Run("restart", func(t *testing.T) {
		t.Parallel()
		cmd := NewRestartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")}))
		got, err := resolveRestartOptions(cmd, &restartOpts{})
		require.NoError(t, err)
		assert.Equal(t, want, got.Env)
	})

	t.Run("reset", func(t *testing.T) {
		t.Parallel()
		cmd := NewResetCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")}))
		got, err := resolveResetOptions(cmd, &resetOpts{})
		require.NoError(t, err)
		assert.Equal(t, want, got.Env)
	})

	// new and run share resolveCreateOptions, covered by
	// TestResolveCreateOptions_EnvFileReachesTheSandboxOptions.

	// The refusal has to travel too: a verb that swallowed the error would start
	// with half an environment.
	t.Run("the conflict error reaches each verb", func(t *testing.T) {
		t.Parallel()
		path := writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")

		startCmd := NewStartCmd()
		require.NoError(t, startCmd.Flags().Parse([]string{"--env", "API_TOKEN=x", "--env-file", path}))
		_, err := resolveStartOptions(startCmd, &startOpts{})
		assertUsageError(t, err, "set by both --env and --env-file")

		restartCmd := NewRestartCmd()
		require.NoError(t, restartCmd.Flags().Parse([]string{"--env", "API_TOKEN=x", "--env-file", path}))
		_, err = resolveRestartOptions(restartCmd, &restartOpts{})
		assertUsageError(t, err, "set by both --env and --env-file")

		resetCmd := NewResetCmd()
		require.NoError(t, resetCmd.Flags().Parse([]string{"--env", "API_TOKEN=x", "--env-file", path}))
		_, err = resolveResetOptions(resetCmd, &resetOpts{})
		assertUsageError(t, err, "set by both --env and --env-file")
	})
}

// TestEnvFlags_RegisteredTogether guards the pair: a verb that takes --env takes
// --env-file, and both are StringArray/String rather than a StringSlice that
// would split a value on commas (DF195).
func TestEnvFlags_RegisteredTogether(t *testing.T) {
	for _, verb := range envVerbs {
		t.Run(verb, func(t *testing.T) {
			t.Parallel()
			cmd := newEnvVerbCmd(t, verb)
			require.NotNil(t, cmd.Flags().Lookup("env"), "--env")
			require.NotNil(t, cmd.Flags().Lookup("env-file"), "--env-file")
			assert.Equal(t, "stringArray", cmd.Flags().Lookup("env").Value.Type())
			assert.Equal(t, "string", cmd.Flags().Lookup("env-file").Value.Type())

			require.NoError(t, cmd.Flags().Parse([]string{"--env", "NO_PROXY=localhost,127.0.0.1"}))
			env, err := cmd.Flags().GetStringArray("env")
			require.NoError(t, err)
			assert.Equal(t, []string{"NO_PROXY=localhost,127.0.0.1"}, env)
		})
	}
}

// TestResolveEnvFromFlags_StdinContention: two flags cannot both read stdin. The
// second reader would see an exhausted stream and silently get nothing.
func TestResolveEnvFromFlags_StdinContention(t *testing.T) {
	for _, promptFlag := range []string{"prompt", "prompt-file"} {
		t.Run(promptFlag, func(t *testing.T) {
			t.Parallel()
			cmd := NewStartCmd()
			require.NoError(t, cmd.Flags().Parse([]string{"--env-file", "-", "--" + promptFlag, "-"}))
			_, err := resolveEnvFromFlags(cmd)
			assertUsageError(t, err, "--env-file - and --"+promptFlag+" - both read stdin")
		})
	}

	t.Run("a prompt from a path is not contention", func(t *testing.T) {
		t.Parallel()
		cmd := NewStartCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", "-", "--prompt-file", "task.md"}))
		cmd.SetIn(strings.NewReader("A=1\n"))
		got, err := resolveEnvFromFlags(cmd)
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"A": "1"}, got, "stdin was not read")
	})

	t.Run("reset has no prompt flags to contend with", func(t *testing.T) {
		t.Parallel()
		cmd := NewResetCmd()
		require.NoError(t, cmd.Flags().Parse([]string{"--env-file", "-"}))
		cmd.SetIn(strings.NewReader("A=1\n"))
		got, err := resolveEnvFromFlags(cmd)
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"A": "1"}, got)
	})
}

// TestResolveCreateOptions_EnvFileReachesTheSandboxOptions follows the value one
// step past the resolver, for the one verb a unit test can reach that far: out of
// the file and into the options the sandbox is actually created with.
// Not parallel, and it establishes its own Layout: resolveCreateOptions resolves
// dir specs through cliutil.Layout(), which panics when the root Layout is unset.
// Relying on another test file in this package to have set it would make this pass
// or fail by test order.
func TestResolveCreateOptions_EnvFileReachesTheSandboxOptions(t *testing.T) {
	clitest.Home(t)
	cmd := NewNewCmd("test")
	require.NoError(t, cmd.Flags().Set("env-file", writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")))

	opts, err := resolveCreateOptions(cmd, "box", ".", nil, "")
	require.NoError(t, err)
	assert.Equal(t, map[string]string{"API_TOKEN": envFileSecret}, opts.Env)
}

// TestEnvFile_WritesNothingToTheLog covers the other half of the exposure: a
// value kept off argv is no better off if the tool writes it down. Nothing on
// this path logs at all — on the success route or either failure route, for any
// verb — and this fails the moment something starts to.
//
// **This is a tripwire, not evidence.** Nothing under resolveEnvFromFlags calls
// slog today, so the empty-buffer assertion holds by construction and cannot go
// red on a revert of the change it accompanies; it goes red when a future edit
// starts logging here. The error-text assertions in the loop are the part that
// measures something. It also sees only this call: logging added further down
// (runStart after the call, the orchestrator, the cli.jsonl sink) is out of its
// reach.
//
// Not parallel: it swaps the default logger.
func TestEnvFile_WritesNothingToTheLog(t *testing.T) {
	var logged bytes.Buffer
	restore := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(restore) })

	good := writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")
	malformed := writeEnvFile(t, envFileSecret+"\n")

	for _, verb := range envVerbs {
		t.Run(verb, func(t *testing.T) {
			cmd := newEnvVerbCmd(t, verb)
			require.NoError(t, cmd.Flags().Parse([]string{"--env-file", good}))
			got, err := resolveEnvFromFlags(cmd)
			require.NoError(t, err)
			require.Equal(t, envFileSecret, got["API_TOKEN"])

			// Both failure routes: the value is within the error's reach on each.
			conflicting := newEnvVerbCmd(t, verb)
			require.NoError(t, conflicting.Flags().Parse([]string{"--env", "API_TOKEN=x", "--env-file", good}))
			_, err = resolveEnvFromFlags(conflicting)
			require.Error(t, err)
			assert.NotContains(t, err.Error(), strings.TrimSpace(envFileSecret))

			unparseable := newEnvVerbCmd(t, verb)
			require.NoError(t, unparseable.Flags().Parse([]string{"--env-file", malformed}))
			_, err = resolveEnvFromFlags(unparseable)
			require.Error(t, err)
			assert.NotContains(t, err.Error(), strings.TrimSpace(envFileSecret))
		})
	}

	assert.NotContains(t, logged.String(), strings.TrimSpace(envFileSecret))
	assert.Empty(t, logged.String(), "the --env-file path logged something")
}
