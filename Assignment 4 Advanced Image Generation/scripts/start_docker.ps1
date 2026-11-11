# Build, start, wait for the trained model, and save actual Docker API evidence.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

function Invoke-DockerCommand {
    param([Parameter(Mandatory = $true)][string[]]$CommandArgs)
    & docker @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed: docker $($CommandArgs -join ' '). Check Docker Desktop and docker compose logs api."
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed. Install Docker Desktop, start it, and run this script again."
}
if (-not (Test-Path -LiteralPath "models/mnist_generator.pt")) {
    throw "models/mnist_generator.pt is missing. Extract the full project archive first."
}

Invoke-DockerCommand -CommandArgs @("compose", "version")
Invoke-DockerCommand -CommandArgs @("info", "--format", "{{.ServerVersion}}")
Invoke-DockerCommand -CommandArgs @("compose", "up", "--build", "-d", "--wait", "--wait-timeout", "180")
Invoke-DockerCommand -CommandArgs @("compose", "exec", "-T", "api", "python", "scripts/check_api.py", "--base-url", "http://127.0.0.1:80", "--output", "/tmp/course-api-evidence/verification.json")

New-Item -ItemType Directory -Path "docker_evidence" -Force | Out-Null
Invoke-DockerCommand -CommandArgs @("compose", "cp", "api:/tmp/course-api-evidence/.", "docker_evidence/")
Invoke-DockerCommand -CommandArgs @("compose", "ps") | Out-File -FilePath "docker_evidence/compose_ps.txt" -Encoding utf8
Invoke-DockerCommand -CommandArgs @("compose", "logs", "--no-color", "api") | Out-File -FilePath "docker_evidence/container.log" -Encoding utf8

$GanPort = if ($env:GAN_PORT) { $env:GAN_PORT } else { "8000" }
Write-Host "Docker API verification passed. Evidence: docker_evidence/"
Write-Host "API docs: http://localhost:$GanPort/docs"
Write-Host "Generated digits: http://localhost:$GanPort/gan/generate.png?count=16&seed=42"
Write-Host "Stop the service when finished: docker compose down"
