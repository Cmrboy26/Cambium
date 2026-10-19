[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$OutputPath,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$source = (Resolve-Path -LiteralPath $Path).Path
if (-not $OutputPath) {
    $OutputPath = [System.IO.Path]::ChangeExtension($source, '.csv')
}
$destination = [System.IO.Path]::GetFullPath($OutputPath)
if ([System.StringComparer]::OrdinalIgnoreCase.Equals($source, $destination)) {
    throw 'The raw recording cannot be used as the output path.'
}
if ((Test-Path -LiteralPath $destination) -and -not $Force) {
    throw 'Output already exists. Choose another path or use -Force.'
}

$number = '[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
$pattern = '^.*?\btick=(?<tick>\d+)\s+input=(?<input>\d+)\s+pos=\[\s*' +
    '(?<x>' + $number + ')d,\s*(?<y>' + $number + ')d,\s*(?<z>' + $number + ')d\]\s+' +
    'rot=\[\s*(?<yaw>' + $number + ')f,\s*(?<pitch>' + $number + ')f\]\s+ground=(?<ground>[01])b\s*$'
$regex = [regex]::new($pattern, [System.Text.RegularExpressions.RegexOptions]::CultureInvariant)
$rows = [System.Collections.Generic.List[object]]::new()
$sourceLine = 0
$ignoredLines = 0
$discontinuities = 0
$seenTicks = [System.Collections.Generic.HashSet[long]]::new()
$duplicates = 0
$previousTick = $null

foreach ($line in [System.IO.File]::ReadLines($source)) {
    $sourceLine++
    if ($line -notmatch '\btick=') {
        $ignoredLines++
        continue
    }
    $sample = $regex.Match($line)
    if (-not $sample.Success) {
        throw "Malformed recorder sample on source line $sourceLine. No CSV was written."
    }
    $tick = [long]::Parse($sample.Groups['tick'].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    $mask = [int]::Parse($sample.Groups['input'].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    if ($mask -lt 0 -or $mask -gt 127) {
        throw "Unknown input-mask bits on source line $sourceLine."
    }
    foreach ($field in @('x', 'y', 'z', 'yaw', 'pitch')) {
        $value = [double]::Parse($sample.Groups[$field].Value,
            [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture)
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
            throw "Non-finite $field on source line $sourceLine."
        }
    }
    if (-not $seenTicks.Add($tick)) { $duplicates++ }
    if ($null -ne $previousTick -and $tick -ne $previousTick + 1) { $discontinuities++ }
    $previousTick = $tick
    $rows.Add([pscustomobject][ordered]@{
        source_line = $sourceLine
        tick = $sample.Groups['tick'].Value
        input = $sample.Groups['input'].Value
        forward = [int](($mask -band 1) -ne 0)
        backward = [int](($mask -band 2) -ne 0)
        left = [int](($mask -band 4) -ne 0)
        right = [int](($mask -band 8) -ne 0)
        jump = [int](($mask -band 16) -ne 0)
        sneak = [int](($mask -band 32) -ne 0)
        sprint = [int](($mask -band 64) -ne 0)
        x = $sample.Groups['x'].Value
        y = $sample.Groups['y'].Value
        z = $sample.Groups['z'].Value
        yaw = $sample.Groups['yaw'].Value
        pitch = $sample.Groups['pitch'].Value
        on_ground = $sample.Groups['ground'].Value
    })
}

if ($rows.Count -eq 0) { throw 'No recorder samples found. No CSV was written.' }
$rows | Export-Csv -LiteralPath $destination -NoTypeInformation -Encoding UTF8
"CSV: $destination"
"Samples: $($rows.Count); ticks: $($rows[0].tick)..$($rows[$rows.Count - 1].tick); ignored lines: $ignoredLines"
"Repeated ticks: $duplicates; non-unit tick transitions: $discontinuities"
'Inputs: ' + (($rows | Group-Object input | Sort-Object { [int]$_.Name } |
    ForEach-Object { "$($_.Name)=$($_.Count) samples" }) -join '; ')
"Airborne samples: $(@($rows | Where-Object on_ground -eq '0').Count)"
'Sampling: server command-block state and last-received inputs; not client-tick-aligned capture.'