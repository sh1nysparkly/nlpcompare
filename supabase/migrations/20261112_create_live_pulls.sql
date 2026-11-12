-- Applied 2026-11-12 via the Supabase MCP (migration name: create_live_pulls).
create table public.live_pulls (
  id uuid primary key default gen_random_uuid(),
  url text not null,
  title text,
  pulled_at timestamptz not null default now(),
  seam text,
  reached_bottom boolean,
  rows jsonb not null,
  pattern_keys jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
comment on table public.live_pulls is 'Page Analysis live pulls: a fetched page chopped into lp_blocks-shaped rows. Versions live in permutations under slug ''pull:<id>''. Never mixed with lp_blocks curation.';
alter table public.live_pulls enable row level security;
create policy anon_all_access on public.live_pulls for all to anon, authenticated using (true) with check (true);
