# Copies the tracked defense files back to their live locations. Run in an elevated PowerShell only after reading
# MANIFEST.md. Does NOT write secrets: config.defense.yaml must be merged by hand with the real secret.
[CmdletBinding(SupportsShouldProcess=$true)]
param([switch]$Tasks)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$map = @{
  "hermes\hooks\guard_tool.py"                  = "$env:USERPROFILE\AppData\Local\hermes\hooks\guard_tool.py"
  "hermes\hooks\claim_audit.py"                 = "$env:USERPROFILE\AppData\Local\hermes\hooks\claim_audit.py"
  "hermes\plugins\loci-rag\__init__.py"         = "$env:USERPROFILE\AppData\Local\hermes\plugins\loci-rag\__init__.py"
  "hermes\plugins\loci-rag\test_selfecho.py"    = "$env:USERPROFILE\AppData\Local\hermes\plugins\loci-rag\test_selfecho.py"
  "hermes\plugins\loci-rag\plugin.yaml"         = "$env:USERPROFILE\AppData\Local\hermes\plugins\loci-rag\plugin.yaml"
  "hermes\SOUL.md"                              = "$env:USERPROFILE\AppData\Local\hermes\SOUL.md"
  "defense-check\check_defense.py"              = "F:\services\defense-check\check_defense.py"
  "defense-check\README.md"                     = "F:\services\defense-check\README.md"
  "litellm\litellm_service.py"                 = "F:\services\litellm\litellm_service.py"
  "litellm\governor\envelope.toml"             = "F:\services\litellm\governor\envelope.toml"
}
foreach ($src in $map.Keys) {
  $from = Join-Path $root $src; $to = $map[$src]
  if ($PSCmdlet.ShouldProcess($to, "restore from $src")) { New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null; Copy-Item $from $to -Force }
}
if ($Tasks) { Get-ChildItem (Join-Path $root "tasks") -Filter *.xml | ForEach-Object { Register-ScheduledTask -Xml (Get-Content $_.FullName -Raw) -TaskName $_.BaseName -Force | Out-Null } }
Write-Host "restored. Merge hermes\config.defense.yaml by hand; set the secret from the owner's store."
