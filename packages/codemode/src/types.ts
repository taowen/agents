import {
  zodToTs,
  printNode as printNodeZodToTs,
  createTypeAlias
} from "zod-to-ts";
import type { ZodType } from "zod";
import type { ToolSet } from "ai";

function toCamelCase(str: string) {
  return str
    .replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())
    .replace(/^[a-z]/, (letter) => letter.toUpperCase());
}

export interface ToolDescriptor {
  description?: string;
  inputSchema: ZodType;
  outputSchema?: ZodType;
}

export type ToolDescriptors = Record<string, ToolDescriptor>;

/**
 * Generate TypeScript type definitions from tool descriptors or an AI SDK ToolSet.
 * These types can be included in tool descriptions to help LLMs write correct code.
 */
export function generateTypes(tools: ToolDescriptors | ToolSet): string {
  let availableTools = "";
  let availableTypes = "";

  for (const [toolName, tool] of Object.entries(tools)) {
    // Handle both our ToolDescriptor and AI SDK Tool types
    const inputSchema =
      "inputSchema" in tool ? tool.inputSchema : tool.parameters;
    const outputSchema = "outputSchema" in tool ? tool.outputSchema : undefined;
    const description = tool.description;

    const inputType = printNodeZodToTs(
      createTypeAlias(
        zodToTs(inputSchema as ZodType, `${toCamelCase(toolName)}Input`).node,
        `${toCamelCase(toolName)}Input`
      )
    );

    const outputType = outputSchema
      ? printNodeZodToTs(
          createTypeAlias(
            zodToTs(outputSchema as ZodType, `${toCamelCase(toolName)}Output`)
              .node,
            `${toCamelCase(toolName)}Output`
          )
        )
      : `type ${toCamelCase(toolName)}Output = unknown`;

    availableTypes += `\n${inputType.trim()}`;
    availableTypes += `\n${outputType.trim()}`;
    availableTools += `\n\t/*\n\t${description?.trim() ?? toolName}\n\t*/`;
    availableTools += `\n\t${toolName}: (input: ${toCamelCase(toolName)}Input) => Promise<${toCamelCase(toolName)}Output>;`;
    availableTools += "\n";
  }

  availableTools = `\ndeclare const codemode: {${availableTools}}`;

  return `
${availableTypes}
${availableTools}
  `.trim();
}
