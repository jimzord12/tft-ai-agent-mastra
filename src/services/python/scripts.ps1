param([string]$Command)

function test {
    uv run python -m unittest discover -s modules -p 'test_*.py'
}

function test-verbose {
    uv run python -m unittest discover -s modules -p 'test_*.py' -v
}

function dev {
    uv run python main.py
}

# If a command argument is provided, execute it
if ($Command) {
    & $Command
} else {
    Write-Host "Available commands: test, test-verbose, dev"
    Write-Host "Usage: . ./scripts.ps1 test"
}