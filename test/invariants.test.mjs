// Regression tests for security-critical, model-independent invariants:
// sandbox path confinement, the read_only tool schema, output capping, and the
// concurrency gate. No network/model needed. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import * as agent from "../dist/agent.js";

const ROOT = "/tmp/local-vibes-test-root";
fs.mkdirSync(ROOT, { recursive: true });

test("path confinement blocks escapes", () => {
  for (const bad of ["../../../etc/passwd", "/etc/passwd", "sub/../../escape", "..", "../sibling"]) {
    assert.throws(() => agent.resolveInSandbox(ROOT, bad), /escapes working_dir/, `not blocked: ${bad}`);
  }
});

test("path confinement allows in-sandbox paths", () => {
  const abs = path.resolve(ROOT); // C:\tmp\... on Windows
  assert.equal(agent.resolveInSandbox(ROOT, "calc.ts"), path.join(abs, "calc.ts"));
  assert.equal(agent.resolveInSandbox(ROOT, "a/b/c.ts"), path.join(abs, "a", "b", "c.ts"));
  assert.equal(agent.resolveInSandbox(ROOT, "."), abs);
});

test("read_only schema excludes mutators", () => {
  const ro = new Set(agent.toolSchemas(true).map((t) => t.function.name));
  for (const m of ["write_file", "edit_file", "run_bash"]) assert.ok(!ro.has(m), `${m} present in read_only`);
  for (const r of ["read_file", "list_dir", "grep", "finish"]) assert.ok(ro.has(r), `${r} missing`);
});

test("read-write schema includes mutators", () => {
  const rw = new Set(agent.toolSchemas(false).map((t) => t.function.name));
  assert.ok(rw.has("write_file") && rw.has("edit_file"), "mutators missing in read-write");
});

test("write_file only creates new files; edit_file inserts text literally", () => {
  const f = "write-once.txt";
  fs.rmSync(path.join(ROOT, f), { force: true });
  assert.match(agent.writeFile(ROOT, f, "a\n"), /^wrote/);
  assert.match(agent.writeFile(ROOT, f, "b\n"), /already exists; use edit_file/);
  assert.equal(fs.readFileSync(path.join(ROOT, f), "utf8"), "a\n");
  agent.editFile(ROOT, f, "a", "$& $1");
  assert.equal(fs.readFileSync(path.join(ROOT, f), "utf8"), "$& $1\n");
});

test("edit_file matches multi-line LF old_string against a CRLF file", () => {
  const f = "crlf.txt";
  fs.writeFileSync(path.join(ROOT, f), "one\r\ntwo\r\nthree\r\n");
  assert.equal(agent.editFile(ROOT, f, "one\ntwo", "uno\ndos"), `edited ${f}`);
  assert.equal(fs.readFileSync(path.join(ROOT, f), "utf8"), "uno\r\ndos\r\nthree\r\n");
});

test("cap truncates oversized output", () => {
  const big = "x".repeat(agent.config.toolResultCap + 500);
  const out = agent.cap(big);
  assert.ok(out.length < big.length && out.includes("truncated"));
  assert.equal(agent.cap("short"), "short");
});

test("semaphore bounds peak concurrency", async () => {
  const sem = new agent.Semaphore(2);
  let active = 0;
  let peak = 0;
  const job = () =>
    sem.run(async () => {
      active++;
      peak = Math.max(peak, active);
      await new Promise((r) => setTimeout(r, 20));
      active--;
    });
  await Promise.all(Array.from({ length: 8 }, job));
  assert.equal(peak, 2, `peak concurrency ${peak} exceeded limit of 2`);
  assert.equal(active, 0, "tasks left active");
});

test("semaphore runs all queued work", async () => {
  const sem = new agent.Semaphore(1);
  const order = [];
  await Promise.all([1, 2, 3].map((n) => sem.run(async () => { order.push(n); })));
  assert.deepEqual(order.sort(), [1, 2, 3]);
});

test("default cohort width is 8 (the mtplx spec/MTP lane)", () => {
  // With LOCAL_VIBES_MAX_CONCURRENCY unset, the default must be 8 — the spec
  // lane's ceiling (8 req x 2 draft/verify rows = the 16-row M1-M16 kernel budget).
  assert.equal(agent.config.maxConcurrency, 8);
});

test("lane maps to the standard OpenAI `user` field (primary mechanism)", () => {
  assert.deepEqual(agent.laneUserField("sess-42"), { user: "sess-42" });
  assert.deepEqual(agent.laneUserField(undefined), {});
  assert.deepEqual(agent.laneUserField(""), {});
});

test("laneHeaders: opt-in, off unless LOCAL_VIBES_LANE_HEADER is set", () => {
  assert.equal(agent.config.laneHeader, undefined); // no header by default
  assert.equal(agent.laneHeaders("sess-42"), undefined);
  assert.equal(agent.laneHeaders(undefined), undefined);
});

test("parseExtraBody: superset passthrough, safe on garbage", () => {
  assert.deepEqual(agent.parseExtraBody('{"top_k":20,"lane":"agent"}'), { top_k: 20, lane: "agent" });
  assert.deepEqual(agent.parseExtraBody(undefined), {}); // pure OpenAI by default
  assert.deepEqual(agent.parseExtraBody(""), {});
  assert.deepEqual(agent.parseExtraBody("not json"), {}); // never throws
  assert.deepEqual(agent.parseExtraBody("[1,2,3]"), {}); // arrays rejected, must be an object
  assert.deepEqual(agent.parseExtraBody("42"), {});
});
