import { promises as fs } from 'node:fs'
import path from 'node:path'
import { NextResponse } from 'next/server'

const modelPaths = {
  'Pure DQN': ['dqn_auv_final.zip', 'dqn_auv_best.zip'],
  'DQN-APF (fixed α)': ['dqn_apf_best.zip', 'dqn_apf_final_final.zip'],
  'Pure PPO': ['ppo_auv_final.zip', 'ppo_auv_best.zip'],
  'PPO-APF (adaptive α)': ['ppo_apf_best.zip', 'ppo_apf_final.zip'],
}

export async function GET() {
  const directory = path.resolve(process.cwd(), '..', 'results', 'models')
  const entries = await Promise.all(Object.entries(modelPaths).map(async ([method, candidates]) => {
    for (const candidate of candidates) {
      try {
        await fs.access(path.join(directory, candidate))
        return [method, { available: true, checkpoint: candidate }] as const
      } catch {
        // Try the next project-standard checkpoint name.
      }
    }
    return [method, { available: false, checkpoint: null }] as const
  }))

  return NextResponse.json({ APF: { available: true, source: 'planning/apf_module.py' }, ...Object.fromEntries(entries) })
}
