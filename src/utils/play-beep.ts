import { spawn } from "node:child_process";

function windowsBeep(frequency: number, duration: number) {
	// Uses .NET console beep in PowerShell
	spawn("powershell", [
		"-Command",
		`[console]::beep(${frequency}, ${duration})`,
	]);
}

/**
 * Plays a beep pattern.
 * @param long - if true, plays one long beep; otherwise, plays two short beeps
 */
export async function playBeepPattern(long: boolean): Promise<void> {
	try {
		if (long) {
			windowsBeep(800, 1000); // 1 second
		} else {
			windowsBeep(800, 200);
			await new Promise((r) => setTimeout(r, 250));
			windowsBeep(800, 200);
		}
	} catch (err) {
		console.error("[Beep Error]", err);
	}
}
