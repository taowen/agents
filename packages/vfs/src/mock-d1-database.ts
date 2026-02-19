/**
 * MockD1Database — in-memory implementation of the Cloudflare D1Database interface.
 *
 * Uses better-sqlite3 under the hood. Implements the subset of D1Database
 * that D1FsAdapter actually uses: prepare, batch.
 */

import Database from "better-sqlite3";
import type BetterSqlite3 from "better-sqlite3";

export class MockD1Database {
  private db: BetterSqlite3.Database;

  constructor() {
    this.db = new Database(":memory:");
    this.db.pragma("journal_mode = WAL");
  }

  /** Execute raw SQL directly (for schema setup in tests). */
  exec(sql: string): void {
    this.db.exec(sql);
  }

  prepare(query: string): MockD1PreparedStatement {
    return new MockD1PreparedStatement(this.db, query);
  }

  async batch(
    statements: MockD1PreparedStatement[]
  ): Promise<{ results: unknown[] }[]> {
    const results: { results: unknown[] }[] = [];
    for (const stmt of statements) {
      results.push(await stmt.run());
    }
    return results;
  }
}

class MockD1PreparedStatement {
  private db: BetterSqlite3.Database;
  private query: string;
  private params: unknown[] = [];

  constructor(db: BetterSqlite3.Database, query: string) {
    this.db = db;
    this.query = query;
  }

  bind(...values: unknown[]): MockD1PreparedStatement {
    this.params = values.map((v) => {
      // Convert Uint8Array to Buffer for better-sqlite3
      if (v instanceof Uint8Array && !(v instanceof Buffer)) {
        return Buffer.from(v.buffer, v.byteOffset, v.byteLength);
      }
      return v;
    });
    return this;
  }

  async first<T = Record<string, unknown>>(
    _colName?: string
  ): Promise<T | null> {
    const stmt = this.db.prepare(this.query);
    const row = stmt.get(...this.params) as Record<string, unknown> | undefined;
    if (!row) return null;
    return this.convertRow(row) as T;
  }

  async all<T = Record<string, unknown>>(): Promise<{ results: T[] }> {
    const stmt = this.db.prepare(this.query);
    const rows = stmt.all(...this.params) as Record<string, unknown>[];
    return { results: rows.map((r) => this.convertRow(r) as T) };
  }

  async run(): Promise<{ results: unknown[] }> {
    const stmt = this.db.prepare(this.query);
    stmt.run(...this.params);
    return { results: [] };
  }

  /** Convert Buffer values in result rows to ArrayBuffer (matching D1 behaviour). */
  private convertRow(row: Record<string, unknown>): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(row)) {
      if (Buffer.isBuffer(value)) {
        // D1 returns ArrayBuffer for blob columns
        result[key] = value.buffer.slice(
          value.byteOffset,
          value.byteOffset + value.byteLength
        );
      } else {
        result[key] = value;
      }
    }
    return result;
  }
}
