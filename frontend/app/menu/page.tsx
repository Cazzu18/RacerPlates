"use client";

import Link from "next/link";

export default function MenuLanding() {
  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-xl text-center space-y-4">
        <h1 className="text-3xl font-semibold">Menu insights</h1>
        <p className="text-sm text-slate-600">
          Use the interactive dashboard to explore the current menu with
          predicted satisfaction scores for each item.
        </p>
        <Link
          href="/dashboard"
          className="inline-flex items-center justify-center rounded bg-black text-white px-4 py-2 text-sm"
        >
          Open interactive dashboard
        </Link>
      </div>
    </main>
  );
}
