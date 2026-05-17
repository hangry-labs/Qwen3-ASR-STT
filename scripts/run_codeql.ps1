param(
    [string]$Database = ".codeql/db",
    [string]$Source = ".codeql/source",
    [string]$Sarif = ".codeql/results.sarif",
    [string]$Suite = "codeql/python-queries:codeql-suites/python-security-and-quality.qls"
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

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $CodeqlRoot | Out-Null

Remove-CodeqlPath -Path $DatabasePath
Remove-CodeqlPath -Path $SourcePath
New-Item -ItemType Directory -Force -Path $SourcePath | Out-Null

$Files = git ls-files --cached --modified --others --exclude-standard | Where-Object {
    $_ -ne "AGENTS.md" -and
    $_ -notlike ".ai/*" -and
    $_ -notlike ".codeql/*"
}

foreach ($File in $Files) {
    $Destination = Join-Path $SourcePath $File
    $DestinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null
    Copy-Item -LiteralPath (Join-Path $RepoRoot $File) -Destination $Destination -Force
}

codeql pack download codeql/python-queries
codeql database create $DatabasePath --language=python --build-mode=none --source-root $SourcePath
codeql database analyze $DatabasePath $Suite --format=sarif-latest --output=$SarifPath

Write-Host "CodeQL SARIF written to $SarifPath"
