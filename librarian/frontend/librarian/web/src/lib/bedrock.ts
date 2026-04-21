import {
  BedrockAgentRuntimeClient,
  RetrieveAndGenerateCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
import type { Citation, RagResponse } from "./types";

function getClient(): BedrockAgentRuntimeClient {
  const region = process.env.AWS_REGION ?? "us-east-1";

  // When running on Vercel / ECS with an IAM role, explicit credentials are
  // not needed — the SDK resolves them from the execution environment.
  const hasExplicitCreds =
    process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY;

  return new BedrockAgentRuntimeClient({
    region,
    ...(hasExplicitCreds
      ? {
          credentials: {
            accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
            ...(process.env.AWS_SESSION_TOKEN
              ? { sessionToken: process.env.AWS_SESSION_TOKEN }
              : {}),
          },
        }
      : {}),
  });
}

interface BedrockRetrievedReference {
  content?: { text?: string };
  location?: {
    type?: string;
    s3Location?: { uri?: string };
    webLocation?: { url?: string };
  };
  metadata?: Record<string, unknown>;
}

interface BedrockCitation {
  retrievedReferences?: BedrockRetrievedReference[];
}

function extractCitations(raw: BedrockCitation[]): Citation[] {
  const seen = new Set<string>();
  const citations: Citation[] = [];

  for (const c of raw) {
    for (const ref of c.retrievedReferences ?? []) {
      const uri =
        ref.location?.s3Location?.uri ??
        ref.location?.webLocation?.url ??
        "";
      if (!uri || seen.has(uri)) continue;
      seen.add(uri);

      const metaTitle =
        (ref.metadata?.["x-amz-bedrock-kb-source-uri-title"] as string) ??
        (ref.metadata?.["title"] as string) ??
        uri.split("/").pop() ??
        uri;

      citations.push({
        url: uri,
        title: String(metaTitle),
        excerpt: ref.content?.text?.slice(0, 200),
      });
    }
  }

  return citations;
}

export async function retrieveAndGenerate(
  query: string,
  sessionId?: string,
): Promise<Omit<RagResponse, "latency_ms" | "backend" | "trace_id">> {
  const knowledgeBaseId = process.env.BEDROCK_KNOWLEDGE_BASE_ID;
  if (!knowledgeBaseId) {
    throw new Error("BEDROCK_KNOWLEDGE_BASE_ID is not configured");
  }

  const modelArn =
    process.env.BEDROCK_MODEL_ARN ??
    `arn:aws:bedrock:${process.env.AWS_REGION ?? "us-east-1"}::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0`;

  const client = getClient();

  const command = new RetrieveAndGenerateCommand({
    input: { text: query },
    retrieveAndGenerateConfiguration: {
      type: "KNOWLEDGE_BASE",
      knowledgeBaseConfiguration: {
        knowledgeBaseId,
        modelArn,
      },
    },
    ...(sessionId ? { sessionId } : {}),
  });

  const result = await client.send(command);

  const responseText = result.output?.text ?? "";
  const citations = extractCitations(
    (result.citations as BedrockCitation[]) ?? [],
  );

  return {
    response: responseText,
    citations,
    confidence_score: null,
    intent: null,
    session_id: result.sessionId,
  };
}
