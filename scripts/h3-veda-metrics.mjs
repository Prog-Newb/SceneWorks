// CPU decoding only. Metrics describe paired output; they never assign quality acceptance.
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, stat } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
const exec = promisify(execFile);
const check = (ok, why) => { if (!ok) throw new Error(`H3 comparison: ${why}`); };
async function receipt(p) { const s = await stat(p), h = createHash("sha256"); for await (const chunk of createReadStream(p)) h.update(chunk); return { path: p, bytes: s.size, sha256: h.digest("hex") }; }
export function mse(bytesA, bytesB, kind = "rgb8") {
  check(bytesA.length > 0 && bytesA.length === bytesB.length && (kind !== "float32" || bytesA.length % 4 === 0), "unaligned/empty decoded arrays");
  const stride = kind === "float32" ? 4 : 1; let sum = 0;
  for (let i = 0; i < bytesA.length; i += stride) { const a = stride === 1 ? bytesA[i] : bytesA.readFloatLE(i), b = stride === 1 ? bytesB[i] : bytesB.readFloatLE(i); check(Number.isFinite(a) && Number.isFinite(b), "nonfinite decoded sample"); sum += (a - b) ** 2; }
  const error = sum / (bytesA.length / stride), range = stride === 1 ? 255 : 2;
  return { samples: bytesA.length / stride, mse: error, psnr: error === 0 ? "Infinity" : 10 * Math.log10(range * range / error), dataRange: range };
}
export async function pairedStreams(a, b, expectedBytes, kind) {
  const ia = a[Symbol.asyncIterator](), ib = b[Symbol.asyncIterator](); let aa = Buffer.alloc(0), bb = Buffer.alloc(0), da = false, db = false, bytes = 0, sum = 0; const stride = kind === "float32" ? 4 : 1;
  while (!da || !db || aa.length || bb.length) {
    if (aa.length < stride && !da) { const r = await ia.next(); da = !!r.done; if (!da) aa = Buffer.concat([aa, Buffer.from(r.value)]); }
    if (bb.length < stride && !db) { const r = await ib.next(); db = !!r.done; if (!db) bb = Buffer.concat([bb, Buffer.from(r.value)]); }
    const n = Math.min(aa.length, bb.length) - Math.min(aa.length, bb.length) % stride;
    if (!n) { check(!(da || db) || (!aa.length && !bb.length && da && db), "decoded stream lengths differ"); if (da && db) break; if ((!da && aa.length < stride) || (!db && bb.length < stride)) continue; throw new Error("H3 comparison: partial sample boundary; decoder stream invalid"); }
    const r = mse(aa.subarray(0, n), bb.subarray(0, n), kind); sum += r.mse * r.samples; bytes += n; aa = aa.subarray(n); bb = bb.subarray(n);
  }
  check(bytes === expectedBytes && bytes > 0, "decoded byte/frame/sample count differs"); const error = sum / (bytes / stride), range = stride === 1 ? 255 : 2;
  return { samples: bytes / stride, mse: error, psnr: error === 0 ? "Infinity" : 10 * Math.log10(range * range / error), dataRange: range };
}
function decode(file, modality, timeout) {
  const args = ["-nostdin", "-hide_banner", "-loglevel", "error", "-hwaccel", "none", "-i", file, "-map", modality === "video" ? "0:v:0" : "0:a:0", ...(modality === "video" ? ["-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo"] : ["-c:a", "pcm_f32le", "-f", "f32le"]), "pipe:1"];
  const p = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"], timeout }); let stderr = ""; p.stderr.on("data", (b) => { stderr = (stderr + b).slice(-4096); });
  const done = new Promise((resolve, reject) => { p.on("error", reject); p.on("close", (code) => code === 0 ? resolve() : reject(new Error(`ffmpeg decode exit ${code}: ${stderr}`))); }); done.catch(() => {});
  return { stream: p.stdout, done, kill: () => p.kill() };
}
export async function collectComparison(dense, sparse, row, outDir, timeout) {
  await mkdir(outDir, { recursive: true });
  const probe = async (file) => JSON.parse((await exec("ffprobe", ["-v", "error", "-count_frames", "-show_streams", "-of", "json", file], { timeout, maxBuffer: 1024 * 1024 })).stdout).streams;
  const streams = await Promise.all([probe(dense.path), probe(sparse.path)]); const [w, h] = row.canvas.split("x").map(Number);
  for (const all of streams) { const v = all.find((s) => s.codec_type === "video"), a = all.find((s) => s.codec_type === "audio"); check(v?.width === w && v.height === h && Number(v.nb_read_frames) === row.frames && v.avg_frame_rate === "24/1" && a?.channels === 2 && Number(a.sample_rate) === 32000, "media geometry/cadence/stereo audio differs"); }
  const compare = async (modality, bytes, kind) => { const a = decode(dense.path, modality, timeout), b = decode(sparse.path, modality, timeout); try { const r = await pairedStreams(a.stream, b.stream, bytes, kind); await Promise.all([a.done, b.done]); return r; } finally { a.kill(); b.kill(); } };
  const video = await compare("video", w * h * 3 * row.frames, "rgb8");
  // Use decoded audio sample counts, not a duration-derived guess. PCM WAV exports retain each arm for audition.
  const audioPaths = [path.join(outDir, "dense.wav"), path.join(outDir, "sparse.wav")];
  for (let i = 0; i < 2; i++) await exec("ffmpeg", ["-nostdin", "-y", "-v", "error", "-hwaccel", "none", "-i", [dense.path, sparse.path][i], "-map", "0:a:0", "-c:a", "pcm_f32le", audioPaths[i]], { timeout, maxBuffer: 1024 * 1024 });
  const audioProbe = await Promise.all(audioPaths.map(probe)); const samples = audioProbe.map((all) => { const a = all.find((s) => s.codec_type === "audio"); return Math.round(Number(a.duration_ts) * Number(a.time_base.split("/")[0]) / Number(a.time_base.split("/")[1]) * 32000); });
  check(samples[0] > 0 && samples[0] === samples[1], "decoded audio sample counts differ");
  const audio = await compare("audio", samples[0] * 2 * 4, "float32");
  const avTiming = streams.map((all) => Object.fromEntries(all.filter((s) => ["video", "audio"].includes(s.codec_type)).map((s) => [s.codec_type, { startTime: Number(s.start_time ?? 0), duration: Number(s.duration) }])));
  check(avTiming.every((x) => Number.isFinite(x.video.duration) && Number.isFinite(x.audio.duration) && Number.isFinite(x.audio.startTime) && Number.isFinite(x.video.startTime)) && avTiming[0].audio.startTime === avTiming[1].audio.startTime && avTiming[0].video.startTime === avTiming[1].video.startTime && avTiming[0].audio.duration === avTiming[1].audio.duration && avTiming[0].video.duration === avTiming[1].video.duration, "audio alignment facts missing/differ");
  const side = path.join(outDir, "side-by-side.mp4");
  await exec("ffmpeg", ["-nostdin", "-y", "-v", "error", "-hwaccel", "none", "-i", dense.path, "-hwaccel", "none", "-i", sparse.path, "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", side], { timeout, maxBuffer: 1024 * 1024 });
  return { video, audio, avTiming, source: "aligned decoded product output; side-by-side export not metric input", sideLabels: ["dense (left)", "sparse (right)"], sideBySide: await receipt(side), denseAudio: await receipt(audioPaths[0]), sparseAudio: await receipt(audioPaths[1]), qualityReview: { reviewer: "Michael", disposition: "pending" } };
}
