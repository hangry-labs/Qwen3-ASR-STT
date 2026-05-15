param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ExpectedModel = "",
    [string]$Container = "",
    [int]$TimeoutSeconds = 600,
    [int]$PollSeconds = 1
)

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$healthUrl = $BaseUrl.TrimEnd("/") + "/health"

while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        $json = $response | ConvertTo-Json -Compress
        Write-Host $json

        if ($ExpectedModel -and $response.model -ne $ExpectedModel) {
            Write-Error "Health endpoint returned model '$($response.model)', expected '$ExpectedModel'."
            exit 2
        }

        if ($response.status -ne "ok") {
            Write-Error "Health endpoint returned status '$($response.status)'."
            exit 3
        }

        exit 0
    } catch {
        if ($Container) {
            $state = docker inspect -f "{{.State.Status}}" $Container 2>$null
            if ($LASTEXITCODE -eq 0 -and $state -and $state -ne "running") {
                Write-Error "Container '$Container' is '$state' before health became ready."
                docker logs --tail 80 $Container
                exit 4
            }
        }

        if ($Container) {
            docker logs --tail 8 $Container 2>$null
        } else {
            Write-Host "Waiting for $healthUrl"
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-Error "Timed out waiting for $healthUrl."
if ($Container) {
    docker logs --tail 120 $Container
}
exit 1
