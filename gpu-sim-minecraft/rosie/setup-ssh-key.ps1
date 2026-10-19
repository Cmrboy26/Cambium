<#
.SYNOPSIS
Install this computer's public SSH key on Rosie and verify key-only login.
.EXAMPLE
.\rosie\setup-ssh-key.ps1 -SshHost 2 -SshUser reissmannc
.EXAMPLE
.\rosie\setup-ssh-key.ps1 -SshHost 2 -SshUser reissmannc -DryRun
#>
[CmdletBinding()]
param(
    [string] $SshHost = $env:ROSIE_SSH_JUMP,
    [string] $SshUser = $env:ROSIE_SSH_USER,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
foreach ($tool in @('ssh', 'ssh-keygen')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required. Install the Windows OpenSSH Client first."
    }
}

if (-not $SshHost) {
    if ($DryRun) { throw 'Specify -SshHost for -DryRun.' }
    $SshHost = Read-Host 'Assigned Rosie management node (1-4), hostname or SSH alias'
}
$SshHost = $SshHost.Trim()
if ($SshHost -match '^[1-4]$') { $SshHost = "dh-mgmt$SshHost.hpc.msoe.edu" }
elseif ($SshHost -match '^dh-mgmt[1-4]$') { $SshHost += '.hpc.msoe.edu' }
if ($SshHost -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw 'Specify a management node (1-4), hostname or SSH alias, without a username.'
}
if (-not $SshUser) {
    $SshUser = if ($env:USERNAME) { $env:USERNAME } else { $env:USER }
    if (-not $DryRun) {
        $answer = (Read-Host "Rosie SSH username [$SshUser]").Trim()
        if ($answer) { $SshUser = $answer }
    }
}
if (-not $SshUser -or $SshUser -notmatch '^[A-Za-z0-9][A-Za-z0-9._@-]*$') {
    throw 'Specify -SshUser for your Rosie account.'
}
if ($SshUser -notmatch '@') { $SshUser += '@ad.msoe.edu' }

$sshDirectory = Join-Path $HOME '.ssh'
$keyPath = Join-Path $sshDirectory 'id_ed25519'
$publicKeyPath = "$keyPath.pub"
$hasPrivateKey = Test-Path -LiteralPath $keyPath -PathType Leaf
$hasPublicKey = Test-Path -LiteralPath $publicKeyPath -PathType Leaf
if (-not $hasPrivateKey -and $hasPublicKey) {
    throw "Found $publicKeyPath without its private key. Restore the matching private key first."
}
if ($DryRun) {
    $keyAction = if ($hasPrivateKey) { 'Reuse' } else { 'Create' }
    Write-Host "$keyAction local key: $keyPath"
    Write-Host "Install only the public key on ${SshUser}@${SshHost}, preserving existing remote keys."
    Write-Host 'Verify key-only login with password prompts disabled. No changes or connections made.'
    return
}

if (-not $hasPrivateKey) {
    New-Item -ItemType Directory -Path $sshDirectory -Force | Out-Null
    Write-Host 'Creating an Ed25519 key. Enter a passphrase directly in the terminal when prompted.'
    & ssh-keygen -t ed25519 -f $keyPath
    if ($LASTEXITCODE -ne 0) { throw 'SSH key generation failed.' }
} elseif (-not $hasPublicKey) {
    $derivedPublicKey = & ssh-keygen -y -f $keyPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not recover the public key from the existing private key.' }
    $derivedPublicKey | Set-Content -LiteralPath $publicKeyPath -Encoding ascii
}

$publicKey = (Get-Content -LiteralPath $publicKeyPath -Raw).Trim()
if ($publicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]*)?$') {
    throw "Unexpected Ed25519 public key format in $publicKeyPath."
}
$encodedPublicKey = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($publicKey))
$sshArguments = @('-T', '-i', $keyPath, '-o', 'IdentitiesOnly=yes',
    '-o', 'ConnectTimeout=20', '-l', $SshUser, $SshHost)
$installScript = @'
set -eu
umask 077
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
key=$(printf '%s' '__PUBLIC_KEY_BASE64__' | base64 -d)
auth="$HOME/.ssh/authorized_keys"
if ! test -f "$auth" || ! grep -qxF -- "$key" "$auth"; then
    if test -f "$auth"; then
        cp -p "$auth" "$auth.backup.$(date +%Y%m%d%H%M%S)"
    fi
    printf '\n%s\n' "$key" >> "$auth"
fi
chmod 600 "$auth"
printf 'Rosie public key installed.\n'
'@

Write-Host "Connect GlobalVPN. Installing the public key on ${SshUser}@${SshHost}."
Write-Host 'Enter any SSH password, passphrase or host-key confirmation only in this terminal.'
$installScript.Replace('__PUBLIC_KEY_BASE64__', $encodedPublicKey).Replace("`r", '') |
    & ssh @sshArguments 'sh -s'
if ($LASTEXITCODE -ne 0) { throw 'Public key installation failed; check GlobalVPN and your SSH account.' }

& ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o PasswordAuthentication=no @sshArguments 'echo ROSIE_KEY_LOGIN_OK'
if ($LASTEXITCODE -ne 0) {
    throw "Key-only login failed. For a passphrase-protected key, load it into your SSH agent with: ssh-add `"$keyPath`", then rerun this script."
}
Write-Host 'Verified: Rosie accepts this key without a password prompt. GlobalVPN is still required.'