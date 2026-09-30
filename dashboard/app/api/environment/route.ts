import { promises as fs } from 'node:fs'
import { execFile } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'
import { NextResponse } from 'next/server'

const execFileAsync = promisify(execFile)

export async function GET() {
  const root = path.resolve(process.cwd(), '..')
  const environment = {
    seed: 42,
    world: { x_max: 100, y_max: 100, z_max: 50, dt: 0.5, max_steps: 500, goal_radius: 3, obstacle_radius: 4, energy_budget_j: 500 },
    start: [5, 5, 5],
    goal: [90, 90, 40],
    obstacles: [] as Array<{ x: number; y: number; z: number; radius: number }>,
  }

  try {
    const snapshotPath = path.join(root, 'open_day_dashboard', 'data', 'environment.json')
    const snapshot = JSON.parse(await fs.readFile(snapshotPath, 'utf8'))
    if (snapshot.obstacles?.length) return NextResponse.json(snapshot)
  } catch {
    // Fall through to the seeded Python environment below.
  }

  try {
    const script = path.join(process.cwd(), 'dashboard_environment.py')
    const { stdout } = await execFileAsync('python', [script], { cwd: root, maxBuffer: 1024 * 1024 })
    return NextResponse.json(JSON.parse(stdout.trim().split('\n').at(-1) ?? '{}'))
  } catch {
    return NextResponse.json(environment)
  }
}
