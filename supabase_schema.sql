create table if not exists public.experiment_sessions (
    participant_id text primary key,
    observer_id text not null,
    completed boolean not null default false,
    updated_at timestamptz not null default timezone('utc', now()),
    session_json jsonb not null
);

create index if not exists experiment_sessions_updated_at_idx
    on public.experiment_sessions (updated_at desc);

create table if not exists public.experiment_batch_counter (
    id integer primary key default 1,
    next_batch integer not null default 1,
    updated_at timestamptz not null default timezone('utc', now()),
    constraint experiment_batch_counter_singleton check (id = 1),
    constraint experiment_batch_counter_next_batch check (next_batch between 1 and 3)
);

insert into public.experiment_batch_counter (id, next_batch)
values (1, 1)
on conflict (id) do nothing;

create or replace function public.allocate_experiment_batch(batch_count integer default 3)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    assigned integer;
begin
    if batch_count < 1 then
        raise exception 'batch_count must be positive';
    end if;

    insert into public.experiment_batch_counter (id, next_batch)
    values (1, 1)
    on conflict (id) do nothing;

    update public.experiment_batch_counter
    set next_batch = case when next_batch >= batch_count then 1 else next_batch + 1 end,
        updated_at = timezone('utc', now())
    where id = 1
    returning case when next_batch = 1 then batch_count else next_batch - 1 end
    into assigned;

    return assigned;
end;
$$;

grant execute on function public.allocate_experiment_batch(integer) to anon, authenticated, service_role;
