import * as screenshot from "screenshot-desktop";

/**
 * Captures a screenshot of the primary display and returns it as a Buffer.
 */
export async function takeScreenshot(): Promise<Buffer> {
	try {
		const img = await screenshot({ format: "png" });
		return img; // This is already a Buffer
	} catch (err) {
		console.error("[Screenshot Error]", err);
		throw new Error("Failed to capture screenshot");
	}
}
