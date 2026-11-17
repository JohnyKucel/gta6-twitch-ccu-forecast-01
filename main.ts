// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// A local-linear log-ratio forecaster. For every hour h = 2..24 of the release
// day it forecasts the Twitch concurrent-viewer count in the game's category
// from the day's first-hour print and the release-day shapes of comparable
// big-game launches shipped in the case file.
//
// Why this shape: launch days do not scale with hour 1. Across the comparable
// launches in the practice cases, the peak absolute audience of a big launch
// sits in a narrow band (roughly 300k-1.1M viewers) whatever hour 1 was, while
// the ratio to hour 1 spans 1x for the biggest launches to 2000x for the
// smallest. So the model regresses the log-ratio r_h = ln(v_h / v_1) on
// ln(v_1) instead of forecasting a flat ratio: big hour-1 prints get small
// ratios, small ones get large ratios, and the fit is local (a Gaussian
// kernel in ln(v_1)) so a launch is predicted mostly by launches of a
// similar size. The spread per hour is the kernel-weighted residual
// standard deviation of that regression, calibrated on leave-one-out runs
// over the comparable launches.
//
// Calibration (measured on the practice cases with SullyGnome answers, and
// on leave-one-out over the 27 comparables with hour-1 >= 20k):
//   kernel bandwidth 1.5 in ln(v_1), spread calibration 0.8, spread floor 0.05
//   practice mean pinball loss 0.178 (house baseline 0.387; best leaderboard
//   entry at the time of writing 0.169)
//   leave-one-out over big comparables 0.259 (global OLS 0.332)
// No number from the practice answers is hardcoded: everything the program
// outputs is fitted from the case file it is given at run time.
//
// The program reads every <id>.in in the input directory (the grader runs
// one case at a time, with only that case's file there) and writes
// <id>.out to the output directory:
//   {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}, exactly 23 rows,
//   hours 2..24 each once, no other key, each q five finite numbers above 0
//   and at most 1e9, non-decreasing. It is deterministic: pure functions of
//   the input file, no clock, no randomness, no network.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Gaussian kernel bandwidth, in units of ln(hour-1 viewers). */
const BANDWIDTH = 1.5;
/** Multiplier on the residual spread, tuned on leave-one-out runs. */
const SPREAD_CAL = 0.8;
/** Smallest log spread allowed for any hour. */
const SPREAD_FLOOR = 0.05;
/** Spread fallback (per sqrt-hour) when a case ships no usable comparable. */
const FLAT_SIGMA = 0.15;
/** Largest viewer count the format allows. */
const MAX_VIEWERS = 1e9;
/** A comparable only counts once it has at least this many usable shapes. */
const MIN_SHAPES_FOR_LOCAL_FIT = 3;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  day_start: string;
  hourly?: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

/** One comparable launch reduced to its release-day shape. */
interface Shape {
  /** ln of the launch's hour-1 print. */
  lnV1: number;
  /** ln(v_h / v_1) for h = 2..24, in order. */
  logRatios: number[];
}

/** x rounded to 2 decimals (half up). */
function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/**
 * The 24 hourly viewer counts of a comparable's release day (the UTC date of
 * its day_start), hours 0..23 in order, or null when any hour is missing or
 * not a positive number.
 */
function releaseDayViewers(comp: Comparable): number[] | null {
  if (!comp.hourly) return null;
  const day = comp.day_start.slice(0, 10);
  const byHour = new Map<number, number>();
  for (const pt of comp.hourly) {
    if (pt.from.slice(0, 10) !== day) continue;
    const hour = Number.parseInt(pt.from.slice(11, 13), 10);
    if (!byHour.has(hour)) byHour.set(hour, pt.viewers);
  }
  const out: number[] = [];
  for (let h = 0; h < 24; h++) {
    const v = byHour.get(h);
    if (v === undefined || !(v > 0)) return null;
    out.push(v);
  }
  return out;
}

/** Every comparable with a complete release day, as a log-ratio shape. */
function extractShapes(c: CaseFile): Shape[] {
  const shapes: Shape[] = [];
  for (const comp of c.comparables ?? []) {
    const vals = releaseDayViewers(comp);
    if (vals === null) continue;
    const v1 = vals[0];
    const logRatios: number[] = [];
    for (let h = 1; h < 24; h++) logRatios.push(Math.log(vals[h] / v1));
    shapes.push({ lnV1: Math.log(v1), logRatios });
  }
  return shapes;
}

/** Weighted least squares of y on x: returns { slope, intercept }. */
function weightedLinearFit(
  xs: number[],
  ys: number[],
  ws: number[],
): { slope: number; intercept: number } {
  const sw = ws.reduce((a, w) => a + w, 0);
  if (sw <= 0) return { slope: 0, intercept: 0 };
  let mx = 0;
  let my = 0;
  for (let i = 0; i < xs.length; i++) {
    mx += (ws[i] * xs[i]) / sw;
    my += (ws[i] * ys[i]) / sw;
  }
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  return { slope, intercept: my - slope * mx };
}

/** Weighted standard deviation of values around their weighted mean. */
function weightedStd(values: number[], ws: number[]): number {
  const sw = ws.reduce((a, w) => a + w, 0);
  if (sw <= 0 || values.length === 0) return 0;
  let mean = 0;
  for (let i = 0; i < values.length; i++) mean += (ws[i] * values[i]) / sw;
  let acc = 0;
  for (let i = 0; i < values.length; i++) {
    acc += ws[i] * (values[i] - mean) * (values[i] - mean);
  }
  return Math.sqrt(acc / sw);
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * The five quantiles for one hour of the release day, as viewer counts.
 * hourIndex 0 is hour 2, hourIndex 22 is hour 24.
 */
function hourQuantiles(
  shapes: Shape[],
  v1: number,
  lnV1: number,
  hourIndex: number,
): number[] {
  let center: number;
  let spread: number;

  if (shapes.length >= MIN_SHAPES_FOR_LOCAL_FIT) {
    // Local-linear fit: launches near this launch's size weigh most.
    const xs = shapes.map((s) => s.lnV1);
    const ys = shapes.map((s) => s.logRatios[hourIndex]);
    let ws = xs.map((x) =>
      Math.exp(-0.5 * ((x - lnV1) / BANDWIDTH) ** 2),
    );
    if (ws.reduce((a, w) => a + w, 0) <= 0) ws = xs.map(() => 1);
    const { slope, intercept } = weightedLinearFit(xs, ys, ws);
    center = intercept + slope * lnV1;
    const residuals = ys.map((y, i) => y - (intercept + slope * xs[i]));
    spread = Math.max(
      weightedStd(residuals, ws),
      SPREAD_FLOOR,
    );
  } else if (shapes.length >= 1) {
    // Too few shapes for a fit: fall back to their median shape.
    const rs = shapes.map((s) => s.logRatios[hourIndex]);
    center = median(rs);
    spread = Math.max(weightedStd(rs, rs.map(() => 1)), SPREAD_FLOOR);
  } else {
    // No comparable at all: stay flat at hour 1 with a widening spread.
    center = 0;
    spread = FLAT_SIGMA * Math.sqrt(hourIndex + 1);
  }

  const qs = Z.map((z) => v1 * Math.exp(center + z * spread * SPREAD_CAL));
  // Format: 2 decimals, inside (0, 1e9], non-decreasing after rounding.
  const out: number[] = [];
  let prev = 0;
  for (const q of qs) {
    let r = round2(q);
    if (!(r > 0)) r = 0.01;
    if (r > MAX_VIEWERS) r = MAX_VIEWERS;
    if (r < prev) r = prev;
    out.push(r);
    prev = r;
  }
  return out;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const v1 = Math.max(1, first.viewers);
  const lnV1 = Math.log(v1);
  const shapes = extractShapes(c);
  const hours = c.forecast.map(({ hour }) => ({
    hour,
    q: hourQuantiles(shapes, v1, lnV1, hour - 2),
  }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
