"use client";
export default function Error({ reset }: { reset: () => void }) {
  return (
    <main className="boot-state">
      <h1>Workspace interrupted</h1>
      <p>The telemetry workspace could not load. Try loading it again.</p>
      <button className="button button-primary" onClick={reset}>
        Reload workspace
      </button>
    </main>
  );
}
