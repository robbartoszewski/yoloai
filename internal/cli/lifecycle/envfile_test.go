// ABOUTME: Tests for --env-file: the parse rules, the stdin form, the refusal of
// ABOUTME: a key supplied twice, and the rule that nothing the file contained is
// ABOUTME: ever echoed — the property that keeps a secret out of logs and reports.
package lifecycle

import (
	"bytes"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// envFileSecret is the value every test passes through --env-file. It carries a
// '#' so a parser that strips inline comments mangles it, and it is the string
// asserted absent from error messages and log output.
const envFileSecret = "s3cret#value "

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
// without quoting it. The error reaches stderr, logs/cli.jsonl and a bug
// report's exit line, and the text it would be quoting is a secret.
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

	t.Run("no --env-file leaves --env alone", func(t *testing.T) {
		t.Parallel()
		got, err := resolveEnv([]string{"PLAIN=1"}, "", nil)
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"PLAIN": "1"}, got)
	})

	t.Run("unreadable file", func(t *testing.T) {
		t.Parallel()
		_, err := resolveEnv(nil, filepath.Join(t.TempDir(), "absent"), nil)
		assertUsageError(t, err, "read --env-file")
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
// the secret is in a file, the command line holds only its path, and the value
// still arrives in the environment the sandbox is given.
func TestEnvFileFlag_ReachesEveryVerb(t *testing.T) {
	for _, verb := range envVerbs {
		t.Run(verb, func(t *testing.T) {
			t.Parallel()
			cmd := newEnvVerbCmd(t, verb)
			path := writeEnvFile(t, "API_TOKEN="+envFileSecret+"\n")
			args := []string{"--env-file", path}
			require.NoError(t, cmd.Flags().Parse(args))

			got, err := resolveEnvFromFlags(cmd)
			require.NoError(t, err)
			assert.Equal(t, map[string]string{"API_TOKEN": envFileSecret}, got)
			assert.NotContains(t, strings.Join(args, " "), strings.TrimSpace(envFileSecret),
				"the value is on the command line, which is what --env-file exists to avoid")
		})
	}
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
func TestResolveCreateOptions_EnvFileReachesTheSandboxOptions(t *testing.T) {
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
