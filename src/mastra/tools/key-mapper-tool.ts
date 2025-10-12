import { createTool } from "@mastra/core/tools";
import * as iohook from "iohook";
import { z } from "zod";
import { playBeepPattern } from "../../utils/play-beep";
import { takeScreenshot } from "../../utils/screenshot";

const RegisterHotkeysSchema = z.object({
	agentUrl: z.string().url().describe("The URL of the agent"),
	mappings: z.record(z.string(), z.string()),
});

export const registerHotkeysTool = createTool({
	id: "register-hotkeys",
	description: "Register global hotkeys and call the associated callback",
	inputSchema: RegisterHotkeysSchema,
	outputSchema: z.object({
		success: z.boolean(),
		message: z.string(),
	}),
	execute: async ({ context }) => {
		const { agentUrl, mappings } = context;

		console.log(`[HotkeyTool] Starting with mappings:`, mappings);

		// Register each hotkey
		for (const combo of Object.keys(mappings)) {
			try {
				// Simple parsing: e.g. "ctrl+alt+s"
				const parts = combo.toLowerCase().split("+");
				const keyPart = parts.pop();
				const mods = parts;

				const keyCode = keyPart.toUpperCase().charCodeAt(0);

				iohook.registerShortcut([keyCode], (keys) => {
					// Check modifiers
					const ctrlOK = !mods.includes("ctrl") || keys.ctrlKey;
					const altOK = !mods.includes("alt") || keys.altKey;
					const shiftOK = !mods.includes("shift") || keys.shiftKey;

					if (ctrlOK && altOK && shiftOK) {
						triggerCallback();
					}
				});

				console.log(`[HotkeyTool] Registered ${combo}`);
			} catch (err) {
				console.error(`[HotkeyTool] Failed to register ${combo}`, err);
			}
		}

		iohook.start();

		async function triggerCallback() {
			try {
				console.log("[HotkeyTool] Hotkey triggered — capturing screenshot");
				const imgBuf = await takeScreenshot();

				const res = await fetch(`${context.agentUrl}/callback`, {
					method: "POST",
					headers: { "Content-Type": "application/octet-stream" },
					body: imgBuf,
				});

				if (!res.ok) {
					console.error(
						"[HotkeyTool] Agent responded with non-OK status",
						res.status,
					);
					await playBeepPattern(false);
					return;
				}

				await playBeepPattern(true);
			} catch (err) {
				console.error("[HotkeyTool] Error during callback or screenshot", err);
				await playBeepPattern(false);
			}
		}

		return {
			success: true,
			message: "Hotkeys registered and listening",
		};
	},
});
