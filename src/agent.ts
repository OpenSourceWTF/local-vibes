/**
 * local-vibes agent core — pure logic + the delegate agent loop.
 *
 * Kept free of MCP/transport concerns so it can be unit-tested and reused.
 * The MCP bin entry (index.ts) is the only file with side effects.
 */
import OpenAI from "openai";
import * as fs from "node:fs";
import * as path from "node:path";
import { spawnSync } from "node:child_process";

type ChatMessage = OpenAI.Chat.Completions.ChatCompletionMessageParam;
type ChatTool = OpenAI.Chat.Completions.ChatCompletionTool;

// ── Config (all overridable via env) ─────────────────────────────────────────
function intEnv(name: string, def: number): number {
  const v = parseInt(process.env[name] ?? "", 10);
  return Number.isFinite(v) ? v : def;
}

/** Parse a JSON object from env; returns {} on missing/invalid/non-object. */
export function parseExtraBody(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {};
  try {
    const v: unknown = JSON.parse(raw);
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export const config = {
  baseURL: process.env.LOCAL_VIBES_BASE_URL ?? "http://localhost:11434/v1",
  apiKey: process.env.LOCAL_VIBES_API_KEY ?? "not-needed", // local servers ignore it
  model: process.env.LOCAL_VIBES_MODEL ?? "qwen3-coder:30b",
  maxIters: intEnv("LOCAL_VIBES_MAX_ITERS", 25),
  temperature: Number.parseFloat(process.env.LOCAL_VIBES_TEMPERATURE ?? "0.1"),
  allowBash: (process.env.LOCAL_VIBES_ALLOW_BASH ?? "1") === "1",
  bashTimeout: intEnv("LOCAL_VIBES_BASH_TIMEOUT", 120), // seconds
  toolResultCap: intEnv("LOCAL_VIBES_TOOL_RESULT_CAP", 12000), // chars
  // Cohort width: how many delegate loops run at once = how many concurrent
  // requests are offered to the backend. On mtplx (PR #200) this picks the lane:
  //   8  → the spec lane's ceiling (8 req × 2 draft/verify rows = the 16-row M1–M16
  //        kernel budget) — keeps MTP speculation AND concurrency. Default.
  //   16 → the AR lane's ceiling (1 row/req, no MTP) — max aggregate throughput.
  // Both run on the tuned M1–M16 kernels; >16 rows falls back to stock MLX.
  // (Ollama: OLLAMA_NUM_PARALLEL; vLLM: max batch.) A batching server
  // admission-queues the overflow, so this is a target, not a hard safety cap.
  maxConcurrency: intEnv("LOCAL_VIBES_MAX_CONCURRENCY", 8),
  // Extra request-body fields merged into every chat.completions call. Empty by
  // default → pure OpenAI. mtplx's server accepts arbitrary extras (extra="allow"):
  // e.g. {"top_k":20} or scheduler/lane/priority hints. Keeping local-vibes a
  // strict superset — generic servers see standard OpenAI, mtplx sees the extras.
  extraBody: parseExtraBody(process.env.LOCAL_VIBES_EXTRA_BODY),
  // Sticky-session lane. When set (env default here, or per-delegate) it is sent
  // as the STANDARD OpenAI `user` field on every request in the session — the
  // idiomatic, portable way to carry a session/affinity id (every OpenAI-compatible
  // server accepts it; cohort-aware servers can pin the session to one lane).
  lane: process.env.LOCAL_VIBES_LANE || undefined,
  // Optional: ALSO send the lane as a custom header, for servers that route on one.
  // Unset by default (the standard `user` field is the primary mechanism).
  laneHeader: process.env.LOCAL_VIBES_LANE_HEADER || undefined,
};

// ── Bounded-concurrency gate (async semaphore) ───────────────────────────────
export class Semaphore {
  private avail: number;
  private waiters: Array<() => void> = [];
  constructor(permits: number) {
    this.avail = Math.max(1, permits);
  }
  get inFlight(): number {
    return this.waiters.length; // queued count (diagnostic)
  }
  private async acquire(): Promise<void> {
    if (this.avail > 0) {
      this.avail--;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
  }
  private release(): void {
    const next = this.waiters.shift();
    if (next) {
      next(); // hand the permit directly to the next waiter
    } else {
      this.avail++;
    }
  }
  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

/** Shared gate limiting how many delegate loops run at once. */
export const gate = new Semaphore(config.maxConcurrency);

// ── Sandbox helpers ──────────────────────────────────────────────────────────
export function cap(text: string): string {
  const limit = config.toolResultCap;
  if (text.length <= limit) return text;
  return text.slice(0, limit) + `\n...[truncated ${text.length - limit} chars]`;
}

/** Resolve `rel` under `root`, refusing anything that escapes the sandbox. */
export function resolveInSandbox(root: string, rel: string): string {
  const rootAbs = path.resolve(root);
  const full = path.resolve(rootAbs, rel);
  if (full !== rootAbs && !full.startsWith(rootAbs + path.sep)) {
    throw new Error(`path ${rel} escapes working_dir`);
  }
  return full;
}

// ── Inner tools the local model may call ─────────────────────────────────────
function readFile(root: string, p: string): string {
  return cap(fs.readFileSync(resolveInSandbox(root, p), "utf8"));
}

export function writeFile(root: string, p: string, content: string): string {
  const full = resolveInSandbox(root, p);
  // Local models tend to retype whole files from memory and silently drop or
  // rewrite code; force targeted edit_file calls on anything that exists.
  if (fs.existsSync(full)) return `ERROR: ${p} already exists; use edit_file to change it`;
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content);
  return `wrote ${content.length} chars to ${p}`;
}

export function editFile(root: string, p: string, oldStr: string, newStr: string): string {
  const full = resolveInSandbox(root, p);
  const text = fs.readFileSync(full, "utf8");
  // Models write "\n"; files checked out on Windows (core.autocrlf) use "\r\n",
  // so multi-line old_strings would never match. Follow the file's line endings.
  if (text.includes("\r\n")) {
    const crlf = (s: string) => s.replace(/\r?\n/g, "\r\n");
    oldStr = crlf(oldStr);
    newStr = crlf(newStr);
  }
  const count = text.split(oldStr).length - 1;
  if (count === 0) return `ERROR: old_string not found in ${p}`;
  if (count > 1) return `ERROR: old_string occurs ${count}× in ${p}; make it unique`;
  fs.writeFileSync(full, text.replace(oldStr, () => newStr)); // no $& / $1 expansion
  return `edited ${p}`;
}

function listDir(root: string, p = "."): string {
  const full = resolveInSandbox(root, p);
  if (!fs.statSync(full).isDirectory()) return `ERROR: ${p} is not a directory`;
  const entries = fs
    .readdirSync(full, { withFileTypes: true })
    .map((d) => d.name + (d.isDirectory() ? "/" : ""))
    .sort();
  return entries.length ? entries.join("\n") : "(empty)";
}

function grep(root: string, pattern: string, p = "."): string {
  const base = resolveInSandbox(root, p);
  const rx = new RegExp(pattern);
  const files: string[] = [];
  const walk = (dir: string): void => {
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      if (d.name === ".git") continue;
      const fp = path.join(dir, d.name);
      if (d.isDirectory()) walk(fp);
      else if (d.isFile()) files.push(fp);
    }
  };
  if (fs.statSync(base).isDirectory()) walk(base);
  else files.push(base);

  const hits: string[] = [];
  for (const fp of files) {
    let content: string;
    try {
      content = fs.readFileSync(fp, "utf8");
    } catch {
      continue;
    }
    const lines = content.split("\n");
    for (let i = 0; i < lines.length; i++) {
      if (rx.test(lines[i])) {
        hits.push(`${path.relative(root, fp)}:${i + 1}: ${lines[i].trim()}`);
        if (hits.length >= 200) return cap(hits.join("\n") + "\n...[capped at 200 hits]");
      }
    }
  }
  return hits.length ? cap(hits.join("\n")) : "(no matches)";
}

function runBash(root: string, command: string): string {
  if (!config.allowBash) return "ERROR: bash is disabled (LOCAL_VIBES_ALLOW_BASH=0)";
  const r = spawnSync("bash", ["-c", command], {
    cwd: root,
    encoding: "utf8",
    timeout: config.bashTimeout * 1000,
    maxBuffer: 8 * 1024 * 1024,
  });
  if (r.error && (r.error as { code?: string }).code === "ETIMEDOUT") {
    return `ERROR: command timed out after ${config.bashTimeout}s`;
  }
  const out = (r.stdout ?? "") + (r.stderr ? `\n[stderr]\n${r.stderr}` : "");
  return cap(`[exit ${r.status}]\n${out}`.trimEnd());
}

const MUTATORS = new Set(["write_file", "edit_file", "run_bash"]);

function dispatch(name: string, root: string, args: Record<string, string>): string {
  switch (name) {
    case "read_file":
      return readFile(root, args.path);
    case "write_file":
      return writeFile(root, args.path, args.content ?? "");
    case "edit_file":
      return editFile(root, args.path, args.old_string ?? "", args.new_string ?? "");
    case "list_dir":
      return listDir(root, args.path ?? ".");
    case "grep":
      return grep(root, args.pattern, args.path ?? ".");
    case "run_bash":
      return runBash(root, args.command ?? "");
    default:
      return `ERROR: unknown tool ${name}`;
  }
}

// ── Tool schemas exposed to the local model ──────────────────────────────────
export function toolSchemas(readOnly: boolean): ChatTool[] {
  const fn = (
    name: string,
    description: string,
    properties: Record<string, unknown>,
    required: string[],
  ): ChatTool => ({
    type: "function",
    function: { name, description, parameters: { type: "object", properties, required } },
  });

  const tools: ChatTool[] = [
    fn("read_file", "Read a file's contents.", { path: { type: "string" } }, ["path"]),
    fn("list_dir", "List the entries of a directory (default: working dir root).", { path: { type: "string" } }, []),
    fn("grep", "Search files under `path` for a regex pattern.", { pattern: { type: "string" }, path: { type: "string" } }, ["pattern"]),
  ];
  if (!readOnly) {
    tools.push(
      fn("write_file", "Create a NEW file with `content` (fails if it exists; use edit_file for existing files).", { path: { type: "string" }, content: { type: "string" } }, ["path", "content"]),
      fn("edit_file", "Replace the unique occurrence of `old_string` with `new_string` in a file.", { path: { type: "string" }, old_string: { type: "string" }, new_string: { type: "string" } }, ["path", "old_string", "new_string"]),
    );
    if (config.allowBash) {
      tools.push(fn("run_bash", "Run a shell command in the working directory and return its output.", { command: { type: "string" } }, ["command"]));
    }
  }
  tools.push(fn("finish", "Call when the task is complete. Provide a concise summary of what you did.", { summary: { type: "string" } }, ["summary"]));
  return tools;
}

// ── The agent loop ───────────────────────────────────────────────────────────
const SYSTEM = `You are an autonomous coding subagent working inside a sandboxed \
working directory. You complete the delegated task by calling the provided tools \
to inspect and modify files. Work iteratively: look before you edit, make the \
smallest change that satisfies the task, and verify your work when possible. Do \
not ask questions — decide and act. When the task is fully done, call \`finish\` \
with a short summary of exactly what you changed. All file paths are relative to \
the working directory root; you cannot access anything outside it.`;

function gitChangeReport(root: string): string {
  const git = (args: string[]): string => {
    const r = spawnSync("git", ["-C", root, ...args], { encoding: "utf8", timeout: 10000 });
    return (r.stdout ?? "").trim();
  };
  const lines: string[] = [];
  const stat = git(["diff", "--stat"]);
  if (stat) lines.push(stat);
  const untracked = git(["ls-files", "--others", "--exclude-standard"]);
  if (untracked) lines.push("new (untracked) files:\n" + untracked);
  return lines.join("\n");
}

export interface RunOptions {
  workingDir?: string;
  readOnly?: boolean;
  /** Per-call sampling temperature; falls back to LOCAL_VIBES_TEMPERATURE. */
  temperature?: number;
  /** Sticky-session lane id; falls back to LOCAL_VIBES_LANE. Sent as a header. */
  lane?: string;
}

/** Standard OpenAI `user` field carrying the sticky-session lane (primary mechanism). */
export function laneUserField(lane: string | undefined): { user?: string } {
  return lane ? { user: lane } : {};
}

/** Optional custom header for the lane — only when a lane AND a header name are set. */
export function laneHeaders(lane: string | undefined): Record<string, string> | undefined {
  return lane && config.laneHeader ? { [config.laneHeader]: lane } : undefined;
}

/**
 * Run one delegate agent loop to completion. Does NOT itself limit concurrency —
 * wrap calls in `gate.run(...)` (index.ts does) to bound how many run at once.
 */
export async function runAgent(task: string, opts: RunOptions = {}): Promise<string> {
  const workingDir = opts.workingDir ?? ".";
  const readOnly = opts.readOnly ?? false;
  const temperature = opts.temperature ?? config.temperature;
  const lane = opts.lane ?? config.lane;
  const root = path.resolve(workingDir);
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
    return `ERROR: working_dir ${JSON.stringify(workingDir)} is not a directory`;
  }

  const client = new OpenAI({ baseURL: config.baseURL, apiKey: config.apiKey });
  const headers = laneHeaders(lane);
  const tools = toolSchemas(readOnly);
  const messages: ChatMessage[] = [
    { role: "system", content: SYSTEM },
    {
      role: "user",
      content: `Working directory root: ${root}\n\nTask:\n${task}\n\nStart by exploring with list_dir/read_file, then act.`,
    },
  ];
  const touched = new Set<string>();

  const finalize = (summary: string, iters: number): string => {
    const laneNote = lane ? `, lane=${lane}` : "";
    const parts = [
      `[local-vibes finished in ${iters} iteration(s)]`,
      `backend: ${config.model} @ ${config.baseURL} (temp=${temperature}${laneNote})`,
    ];
    if (touched.size) parts.push("files touched: " + [...touched].sort().join(", "));
    const changes = readOnly ? "" : gitChangeReport(root);
    if (changes) parts.push("git changes:\n" + changes);
    parts.push("\nsummary:\n" + (summary || "(none)"));
    return parts.join("\n");
  };

  for (let i = 1; i <= config.maxIters; i++) {
    let msg: OpenAI.Chat.Completions.ChatCompletionMessage;
    try {
      const resp = await client.chat.completions.create(
        {
          model: config.model,
          messages,
          tools,
          tool_choice: "auto",
          temperature,
          ...laneUserField(lane),
          ...config.extraBody,
        } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
        headers ? { headers } : undefined,
      );
      msg = resp.choices[0].message;
    } catch (e) {
      return `ERROR calling backend ${config.model} @ ${config.baseURL}: ${e}`;
    }

    messages.push({
      role: "assistant",
      content: msg.content ?? null,
      tool_calls: msg.tool_calls,
    });

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      return finalize(msg.content ?? "(model returned no content)", i);
    }

    for (const tc of msg.tool_calls) {
      const call = tc as OpenAI.Chat.Completions.ChatCompletionMessageToolCall & {
        function: { name: string; arguments: string };
      };
      const name = call.function.name;
      let args: Record<string, string> = {};
      try {
        args = JSON.parse(call.function.arguments || "{}");
      } catch {
        args = {};
      }
      if (name === "finish") return finalize(args.summary ?? "", i);

      let result: string;
      try {
        if (readOnly && MUTATORS.has(name)) result = `ERROR: ${name} is disabled in read_only mode`;
        else result = dispatch(name, root, args);
        if (MUTATORS.has(name) && !result.startsWith("ERROR") && args.path) touched.add(args.path);
      } catch (e) {
        result = `ERROR in ${name}: ${e}`;
      }
      messages.push({ role: "tool", tool_call_id: call.id, content: cap(result) });
    }
  }
  return finalize("(reached max iterations without calling finish)", config.maxIters);
}
