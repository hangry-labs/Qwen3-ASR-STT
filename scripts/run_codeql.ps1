param(
    [string]$Database = ".codeql/db",
    [string]$Source = ".codeql/source",
    [string]$Sarif = ".codeql/results.sarif",
    [string]$QueryPackVersion = "1.8.2",
    [string]$Suite = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CodeqlRoot = Join-Path $RepoRoot ".codeql"
$DatabasePath = Join-Path $RepoRoot $Database
$SourcePath = Join-Path $RepoRoot $Source
$SarifPath = Join-Path $RepoRoot $Sarif

function Remove-CodeqlPath {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $Resolved = (Resolve-Path -LiteralPath $Path).Path
    $AllowedRoot = (Resolve-Path -LiteralPath $CodeqlRoot).Path
    if (-not $Resolved.StartsWith($AllowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside .codeql: $Resolved"
    }

    Remove-Item -LiteralPath $Resolved -Recurse -Force
}

function Invoke-Codeql {
    & codeql @args
    if ($LASTEXITCODE -ne 0) {
        throw "CodeQL command failed with exit code ${LASTEXITCODE}: codeql $($args -join ' ')"
    }
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $CodeqlRoot | Out-Null

Remove-CodeqlPath -Path $DatabasePath
Remove-CodeqlPath -Path $SourcePath
New-Item -ItemType Directory -Force -Path $SourcePath | Out-Null

$Files = git ls-files --cached --modified --others --exclude-standard | Where-Object {
    $_ -ne "AGENTS.md" -and
    $_ -notlike ".ai/*" -and
    $_ -notlike ".codeql/*" -and
    (Test-Path -LiteralPath (Join-Path $RepoRoot $_) -PathType Leaf)
}

foreach ($File in $Files) {
    $Destination = Join-Path $SourcePath $File
    $DestinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null
    Copy-Item -LiteralPath (Join-Path $RepoRoot $File) -Destination $Destination -Force
}

Invoke-Codeql pack download "codeql/python-queries@$QueryPackVersion"

if (-not $Suite) {
    $Suite = Join-Path $HOME ".codeql/packages/codeql/python-queries/$QueryPackVersion/codeql-suites/python-security-and-quality.qls"
}
if (-not (Test-Path -LiteralPath $Suite -PathType Leaf)) {
    throw "CodeQL query suite not found: $Suite"
}

Invoke-Codeql database create $DatabasePath --language=python --build-mode=none --source-root $SourcePath
Invoke-Codeql database analyze $DatabasePath $Suite --format=sarif-latest --output=$SarifPath

$Results = @((Get-Content -LiteralPath $SarifPath -Raw | ConvertFrom-Json).runs[0].results)
if ($Results.Count -gt 0) {
    throw "CodeQL reported $($Results.Count) finding(s). Review $SarifPath."
}

Write-Host "CodeQL completed with no findings. SARIF written to $SarifPath"
