import { promises as fs } from 'node:fs'
import path from 'node:path'
import { NextResponse } from 'next/server'

export async function GET() {
  const file = path.resolve(process.cwd(), '..', 'results', 'results_1788746341.json')
  const data = JSON.parse(await fs.readFile(file, 'utf8'))
  return NextResponse.json(data)
}
