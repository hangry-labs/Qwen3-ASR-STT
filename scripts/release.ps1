param(
    [string]$DryRun = "0",
    [string]$NextVersion = "",
    [string]$SkipValidation = "0",
    [string]$Image = "qwen3-asr-stt:test"
)

$ErrorActionPreference = "Stop"

function Test-Enabled {
    param([string]$Value)
    return $Value -match '^(1|true|yes|y)$'
}

function Set-Utf8Text {
    param([string]$Path, [string]$Text)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($resolved, $Text, $encoding)
}

function Convert-ToPackageVersion {
    param([string]$Version)
    if ($Version -match '^\d+\.\d+$') { return "$Version.0" }
    if ($Version -match '^\d+\.\d+\.\d+$') { return $Version }
    throw "Version '$Version' must look like 0.2 or 0.2.0."
}

function Get-NextMinorSnapshot {
    param([string]$Version)
    if ($Version -notmatch '^(\d+)\.(\d+)\.\d+$') {
        throw "Cannot infer the next snapshot from '$Version'. Pass NEXT_VERSION=..."
    }
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2] + 1
    return "$major.$minor.0-snapshot"
}

function Invoke-Native {
    param([string]$Description, [scriptblock]$Action)
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Step {
    param([string]$Description, [scriptblock]$Action)
    Write-Host "==> $Description"
    if (-not (Test-Enabled $DryRun)) {
        & $Action
    }
}

function Get-ProjectVersion {
    $content = Get-Content -Raw -LiteralPath "pyproject.toml"
    $match = [regex]::Match($content, '(?m)^version = "([^"]+)"(?=\r?$)')
    if (-not $match.Success) {
        throw "Could not read [project].version from pyproject.toml."
    }
    return $match.Groups[1].Value
}

function Set-ProjectVersion {
    param([string]$Version)
    $content = Get-Content -Raw -LiteralPath "pyproject.toml"
    $pattern = [regex]::new('(?m)^version = "[^"]+"(?=\r?$)')
    $updated = $pattern.Replace($content, "version = `"$Version`"", 1)
    if ($updated -eq $content -and (Get-ProjectVersion) -ne $Version) {
        throw "Failed to update [project].version in pyproject.toml."
    }
    Set-Utf8Text "pyproject.toml" $updated
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

Write-Host "Release automation may create local commits and an annotated tag."
Write-Host "It never pushes Git refs or Docker images."

if (-not (Test-Path -LiteralPath "VERSION")) {
    throw "VERSION file is missing from the repository root."
}

$currentVersion = (Get-Content -Raw -LiteralPath "VERSION").Trim()
$versionMatch = [regex]::Match($currentVersion, '^(\d+\.\d+(?:\.\d+)?)(-snapshot)?$')
if (-not $versionMatch.Success) {
    throw "VERSION must look like 0.1.0 or 0.2.0-snapshot. Current: '$currentVersion'"
}

$releaseVersion = Convert-ToPackageVersion $versionMatch.Groups[1].Value
$isSnapshot = $versionMatch.Groups[2].Success
$releaseTag = "v$releaseVersion"
$expectedProjectVersion = if ($isSnapshot) { "$releaseVersion.dev0" } else { $releaseVersion }
$projectVersion = Get-ProjectVersion

if ($projectVersion -ne $expectedProjectVersion) {
    throw "pyproject.toml version '$projectVersion' does not match VERSION '$currentVersion' (expected '$expectedProjectVersion')."
}

if ([string]::IsNullOrWhiteSpace($NextVersion)) {
    $nextSnapshotVersion = Get-NextMinorSnapshot $releaseVersion
} else {
    $nextSnapshotVersion = $NextVersion.Trim()
}

$nextMatch = [regex]::Match($nextSnapshotVersion, '^(\d+\.\d+(?:\.\d+)?)-snapshot$')
if (-not $nextMatch.Success) {
    throw "NEXT_VERSION must look like 0.2.0-snapshot. Current: '$nextSnapshotVersion'"
}

$nextReleaseVersion = Convert-ToPackageVersion $nextMatch.Groups[1].Value
if ([version]$nextReleaseVersion -le [version]$releaseVersion) {
    throw "NEXT_VERSION '$nextSnapshotVersion' must be newer than '$releaseVersion'."
}
$nextProjectVersion = "$nextReleaseVersion.dev0"

$readme = Get-Content -Raw -LiteralPath "README.md"
$stableHeading = "### $releaseTag"
$snapshotHeading = "### v$currentVersion"
if (-not $readme.Contains($stableHeading) -and -not $readme.Contains($snapshotHeading)) {
    throw "README.md must contain a '$stableHeading' release-history heading before release."
}

$branch = (git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not determine the current Git branch."
}
if ($branch -ne "main") {
    throw "Releases must run from main. Current branch: '$branch'"
}

$status = git status --porcelain --untracked-files=all -- . ":(exclude).ai" ":(exclude).ai/**" ":(exclude)AGENTS.md"
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the Git working tree."
}
if ($status) {
    if (Test-Enabled $DryRun) {
        Write-Warning "The real release will require a clean working tree outside .ai/ and AGENTS.md."
        $status | ForEach-Object { Write-Host "  $_" }
    } else {
        throw "Working tree outside .ai/ and AGENTS.md must be clean before release. Commit or stash the listed changes first.`n$($status -join "`n")"
    }
}

if (-not (Test-Enabled $DryRun)) {
    Invoke-Native "Fetch origin/main and tags" { git fetch origin main --tags }
    $head = (git rev-parse HEAD).Trim()
    $originMain = (git rev-parse refs/remotes/origin/main).Trim()
    if ($head -ne $originMain) {
        throw "main must be synchronized with origin/main before release. HEAD=$head origin/main=$originMain"
    }
}

if (git tag --list $releaseTag) {
    throw "Tag $releaseTag already exists."
}

Write-Host "Release version: $releaseVersion"
Write-Host "Release tag:     $releaseTag"
Write-Host "Package version: $releaseVersion"
Write-Host "Next snapshot:   $nextSnapshotVersion"
Write-Host "Next package:    $nextProjectVersion"
Write-Host "Validation:      $(if (Test-Enabled $SkipValidation) { 'skipped' } else { 'compile, CodeQL, Dockerfile, baked image, containerized unit tests' })"

Invoke-Step "Run release validation" {
    if (-not (Test-Enabled $SkipValidation)) {
        $venvPython = Join-Path $root ".venv\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $venvPython)) {
            throw "Release validation requires $venvPython."
        }
        Invoke-Native "Python compilation" { & $venvPython -m compileall -q qwen_asr tests testbench examples }
        Invoke-Native "CodeQL analysis" { task codeql }
        Invoke-Native "Dockerfile validation" { docker build --check . }
        Invoke-Native "Baked image build" { task image "IMAGE=$Image" }
        $testsPath = (Resolve-Path -LiteralPath "tests").Path
        Invoke-Native "Containerized unit tests" {
            docker run --rm --entrypoint python -v "${testsPath}:/app/tests:ro" $Image -m unittest discover -s /app/tests -v
        }
    }
}

Invoke-Step "Update release metadata for $releaseTag" {
    Set-Utf8Text "VERSION" "$releaseVersion`n"
    Set-ProjectVersion $releaseVersion

    foreach ($doc in @("README.md", "docs/dockerhub.md")) {
        if (-not (Test-Path -LiteralPath $doc)) { continue }
        $content = Get-Content -Raw -LiteralPath $doc
        $content = $content.Replace("### v$currentVersion", $stableHeading)
        $content = $content.Replace(":v$currentVersion", ":$releaseTag")
        Set-Utf8Text $doc $content
    }

    $updatedReadme = Get-Content -Raw -LiteralPath "README.md"
    if (-not $updatedReadme.Contains($stableHeading)) {
        throw "README.md does not contain the required '$stableHeading' release-history heading."
    }
}

Invoke-Step "Commit release metadata when needed and tag $releaseTag" {
    $releaseFiles = @("VERSION", "pyproject.toml", "README.md", "docs/dockerhub.md")
    $releaseChanges = git status --porcelain -- $releaseFiles
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect release metadata changes."
    }
    if ($releaseChanges) {
        Invoke-Native "Stage release metadata" { git add -- $releaseFiles }
        Invoke-Native "Create release commit" { git commit -m "release: $releaseTag" }
    } else {
        Write-Host "Release metadata is already committed; tagging the current HEAD."
    }
    Invoke-Native "Create annotated release tag" { git tag -a $releaseTag -m "Release $releaseTag" }
}

Invoke-Step "Prepare $nextSnapshotVersion" {
    Set-Utf8Text "VERSION" "$nextSnapshotVersion`n"
    Set-ProjectVersion $nextProjectVersion
    Invoke-Native "Stage next snapshot metadata" { git add -- VERSION pyproject.toml }
    Invoke-Native "Create next snapshot commit" { git commit -m "chore: start $nextSnapshotVersion" }
}

if (Test-Enabled $DryRun) {
    Write-Host "Dry run complete. No files, commits, tags, or remote refs were changed."
} else {
    Write-Host "Release workflow complete. No push was performed."
    Write-Host "Review the local commits and $releaseTag before pushing them."
}
