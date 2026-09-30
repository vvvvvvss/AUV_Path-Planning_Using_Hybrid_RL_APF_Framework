import { execFile } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'
import { NextRequest, NextResponse } from 'next/server'

const execFileAsync = promisify(execFile)
const allowed = new Set(['APF', 'Pure DQN', 'Pure PPO', 'PPO-APF (adaptive α)', 'DQN-APF (fixed α)'])

export async function GET(request: NextRequest) {
  const method = request.nextUrl.searchParams.get('method') ?? 'APF'
  const seed = Number(request.nextUrl.searchParams.get('seed') ?? '42')
  if (!allowed.has(method) || !Number.isInteger(seed)) {
    return NextResponse.json({ error: 'Unsupported method or seed' }, { status: 400 })
  }

  const script = path.join(process.cwd(), 'dashboard_simulation.py')
  const root = path.resolve(process.cwd(), '..')
  try {
    const { stdout } = await execFileAsync('python', [script, method, String(seed)], { cwd: root, maxBuffer: 8 * 1024 * 1024 })
    const output = stdout.trim().split('\n').at(-1)
    return NextResponse.json({ method, seed, ...(output ? JSON.parse(output) : {}) })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Simulation failed'
    return NextResponse.json({ error: message, method, seed }, { status: 503 })
  }
}
