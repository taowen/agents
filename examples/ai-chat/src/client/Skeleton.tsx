export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded bg-kumo-elevated ${className ?? ""}`}
    />
  );
}

export function FormFieldSkeleton() {
  return (
    <div className="space-y-1">
      <Skeleton className="h-3 w-20" />
      <Skeleton className="h-9 w-full" />
    </div>
  );
}

export function SessionListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-1 px-2">
      {Array.from({ length: count }, (_, i) => (
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  );
}

export function AppShellSkeleton() {
  return (
    <div className="flex h-screen">
      <div className="hidden md:block w-64 border-r border-kumo-line p-3">
        <Skeleton className="h-9 w-full mb-3" />
        <SessionListSkeleton />
      </div>
      <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8">
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-4 w-32" />
      </div>
    </div>
  );
}
