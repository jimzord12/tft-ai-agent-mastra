import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { playBeepPattern } from "../../utils/play-beep";

export const beepTool = createTool({
	id: "beepTool",
	description: "Play a long or short beep in Windows",
	inputSchema: z.object({
		long: z.boolean(),
	}),
	outputSchema: z.object({
		success: z.boolean(),
	}),
	execute: async ({ context }) => {
		const { long } = context;
		await playBeepPattern(long);
		return { success: true };
	},
});
