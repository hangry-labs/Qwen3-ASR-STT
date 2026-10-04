param(
    [string]$DryRun = "0",
    [string]$NextVersion = "",
    [string]$SkipValidation = "0"
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
    if ($Version -match '^(\d+)\.(\d+)$') {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2] + 1
        return "$major.$minor-snapshot"
    }
    if ($Version -match '^(\d+)\.(\d+)\.\d+$') {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2] + 1
        return "$major.$minor-snapshot"
    }
    throw "Cannot infer the next snapshot from '$Version'. Pass NEXT_VERSION=..."
}

function Get-VersionHistoryDockerSection {
    param(
        [string]$Version,
        [string]$AvailabilityLine,
        [string]$LineEnding,
        [bool]$Rolling = $false
    )

    $containerVersion = $Version.Replace(".", "-")
    $standardContainer = if ($Rolling) { "qwen3-asr-stt" } else { "qwen3-asr-stt-v$containerVersion" }
    $tinyContainer = if ($Rolling) { "qwen3-asr-stt-tiny" } else { "qwen3-asr-stt-v$containerVersion-tiny" }
    $standardTag = if ($Rolling) { "latest" } else { "v$Version" }
    $tinyTag = if ($Rolling) { "latest_tiny" } else { "v${Version}_tiny" }
    return @(
        $AvailabilityLine,
        "",
        "**Standard image**",
        "",
        '```bash',
        "docker run --name $standardContainer --restart unless-stopped -p 8000:8000 --gpus all -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:$standardTag",
        '```',
        "",
        "**Tiny image**",
        "",
        '```bash',
        "docker run --name $tinyContainer --restart unless-stopped -p 8000:8000 --gpus all -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:$tinyTag",
        '```'
    ) -join $LineEnding
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

Write-Host "Release automation creates local commits and an annotated tag, then pushes them atomically."
Write-Host "Docker images are published only by GitHub Actions."

if (-not (Test-Path -LiteralPath "VERSION")) {
    throw "VERSION file is missing from the repository root."
}

$currentVersion = (Get-Content -Raw -LiteralPath "VERSION").Trim()
$versionMatch = [regex]::Match($currentVersion, '^(\d+\.\d+(?:\.\d+)?)-snapshot$')
if (-not $versionMatch.Success) {
    throw "VERSION must be a snapshot such as 1.0-snapshot or 1.0.0-snapshot before release. Current: '$currentVersion'"
}

$releaseVersion = Convert-ToPackageVersion $versionMatch.Groups[1].Value
$releaseDisplayVersion = $versionMatch.Groups[1].Value
$releaseTag = "v$releaseDisplayVersion"
$expectedProjectVersion = "$releaseVersion.dev0"
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
    throw "NEXT_VERSION must look like 1.1-snapshot or 1.1.0-snapshot. Current: '$nextSnapshotVersion'"
}

$nextReleaseVersion = Convert-ToPackageVersion $nextMatch.Groups[1].Value
if ([version]$nextReleaseVersion -le [version]$releaseVersion) {
    throw "NEXT_VERSION '$nextSnapshotVersion' must be newer than '$releaseVersion'."
}
$nextProjectVersion = "$nextReleaseVersion.dev0"
$nextDisplayVersion = $nextMatch.Groups[1].Value
$nextDevelopmentHeading = "### v$nextDisplayVersion Snapshot"
$nextDevelopmentHeadingPattern = '(?m)^' + [regex]::Escape($nextDevelopmentHeading) + '\r?$'
$developmentImageNotice = 'The current development snapshot is published through the rolling tags from `main`:'
$stableImageNotice = "Run this release with either image variant:"

$readme = Get-Content -Raw -LiteralPath "README.md"
$stableHeading = "### $releaseTag"
$snapshotHeading = "### v$releaseDisplayVersion Snapshot"
$legacySnapshotHeading = "### v$currentVersion"
$developmentHeading = "### v$releaseVersion (in development)"
$stableHeadingPattern = '(?m)^' + [regex]::Escape($stableHeading) + '\r?$'
$snapshotHeadingPattern = '(?m)^(' +
    [regex]::Escape($snapshotHeading) + '|' +
    [regex]::Escape($legacySnapshotHeading) + '|' +
    [regex]::Escape($developmentHeading) + ')\r?$'
if (
    -not [regex]::IsMatch($readme, $stableHeadingPattern) -and
    -not [regex]::IsMatch($readme, $snapshotHeadingPattern)
) {
    throw "README.md must contain an exact '$stableHeading' or '$snapshotHeading' release-history heading before release."
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

Write-Host "Release version: $releaseDisplayVersion"
Write-Host "Release tag:     $releaseTag"
Write-Host "Package version: $releaseVersion"
Write-Host "Next snapshot:   $nextSnapshotVersion"
Write-Host "Next package:    $nextProjectVersion"
Write-Host "Validation:      $(if (Test-Enabled $SkipValidation) { 'skipped' } else { 'metadata, compile, CodeQL, Dockerfile' })"

Invoke-Step "Run release validation" {
    if (-not (Test-Enabled $SkipValidation)) {
        $venvPython = Join-Path $root ".venv\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $venvPython)) {
            throw "Release validation requires $venvPython."
        }
        Invoke-Native "Python compilation" { & $venvPython -m compileall -q qwen_asr tests testbench }
        Invoke-Native "CodeQL analysis" { task codeql }
        Invoke-Native "Dockerfile validation" { docker build --check . }
    }
}

Invoke-Step "Update release metadata for $releaseTag" {
    Set-Utf8Text "VERSION" "$releaseDisplayVersion`n"
    Set-ProjectVersion $releaseVersion

    foreach ($doc in @("README.md", "docs/dockerhub.md")) {
        if (-not (Test-Path -LiteralPath $doc)) { continue }
        $content = Get-Content -Raw -LiteralPath $doc
        $content = $content.Replace($snapshotHeading, $stableHeading)
        $content = $content.Replace($legacySnapshotHeading, $stableHeading)
        $content = $content.Replace($developmentHeading, $stableHeading)
        if ($doc -eq "README.md") {
            $lineEnding = if ($content.Contains("`r`n")) { "`r`n" } else { "`n" }
            $developmentDockerSection = Get-VersionHistoryDockerSection `
                -Version $releaseDisplayVersion `
                -AvailabilityLine $developmentImageNotice `
                -LineEnding $lineEnding `
                -Rolling $true
            $stableDockerSection = Get-VersionHistoryDockerSection `
                -Version $releaseDisplayVersion `
                -AvailabilityLine $stableImageNotice `
                -LineEnding $lineEnding
            if (-not $content.Contains($developmentDockerSection)) {
                throw "README.md does not contain the expected rolling Docker section for $developmentHeading."
            }
            $content = $content.Replace($developmentDockerSection, $stableDockerSection)
        }
        $content = $content.Replace(":v$releaseDisplayVersion", ":$releaseTag")
        Set-Utf8Text $doc $content
    }

    $updatedReadme = Get-Content -Raw -LiteralPath "README.md"
    if (-not [regex]::IsMatch($updatedReadme, $stableHeadingPattern)) {
        throw "README.md does not contain the required exact '$stableHeading' release-history heading."
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

    $updatedReadme = Get-Content -Raw -LiteralPath "README.md"
    if (-not [regex]::IsMatch($updatedReadme, $nextDevelopmentHeadingPattern)) {
        $stableHeadingMatches = [regex]::Matches($updatedReadme, $stableHeadingPattern)
        if ($stableHeadingMatches.Count -ne 1) {
            throw "README.md must contain exactly one '$stableHeading' heading before preparing the next snapshot."
        }

        $lineEnding = if ($updatedReadme.Contains("`r`n")) { "`r`n" } else { "`n" }
        $stableHeadingLine = "$stableHeading$lineEnding"
        if (-not $updatedReadme.Contains($stableHeadingLine)) {
            throw "README.md release heading '$stableHeading' is not followed by a line ending."
        }

        $nextDockerSection = Get-VersionHistoryDockerSection `
            -Version $nextDisplayVersion `
            -AvailabilityLine $developmentImageNotice `
            -LineEnding $lineEnding `
            -Rolling $true
        $nextHistorySection =
            "$nextDevelopmentHeading$lineEnding$lineEnding" +
            "- No changes yet.$lineEnding$lineEnding" +
            "$nextDockerSection$lineEnding$lineEnding" +
            $stableHeadingLine
        $updatedReadme = $updatedReadme.Replace($stableHeadingLine, $nextHistorySection)
        Set-Utf8Text "README.md" $updatedReadme
    }

    Invoke-Native "Stage next snapshot metadata" { git add -- VERSION pyproject.toml README.md }
    Invoke-Native "Create next snapshot commit" { git commit -m "chore: start $nextSnapshotVersion" }
}

Invoke-Step "Push main and $releaseTag atomically" {
    Invoke-Native "Push release commits and tag" { git push --atomic origin main $releaseTag }
}

if (Test-Enabled $DryRun) {
    Write-Host "Dry run complete. No files, commits, tags, or remote refs were changed."
    Write-Host "The public GitHub Release entry remains a manual step after tagged deployment validation."
} else {
    Write-Host "Release workflow complete. main and $releaseTag were pushed atomically."
    Write-Host "GitHub Actions is responsible for publishing the release images."
    Write-Host "After validating the tagged deployment, create the public GitHub Release entry manually."
}
