package net.cambium;

import java.util.UUID;
import java.util.concurrent.CompletableFuture;

final class BridgeInference<T> {
	static final int MAX_TICK_AGE = 2;
	static final long MAX_NANOS = 250_000_000L;
	static final int MAX_STALE = 5;

	record Result<T>(T value, int tick, String epoch) {}

	private String epoch = UUID.randomUUID().toString();
	private CompletableFuture<T> pending;
	private int sentTick;
	private long sentNanos;
	private int stale;

	String epoch() { return epoch; }
	boolean pending() { return pending != null; }
	int stale() { return stale; }

	void submit(CompletableFuture<T> request, int tick, long nanos) {
		if (request == null || tick < 0 || pending != null) throw new IllegalStateException("Invalid Bridge inference submission");
		pending = request;
		sentTick = tick;
		sentNanos = nanos;
	}

	Result<T> poll(int tick, long nanos) {
		if (pending == null || !pending.isDone()) return null;
		CompletableFuture<T> completed = pending;
		pending = null;
		T value = completed.join();
		if (tick < sentTick || nanos < sentNanos || tick - sentTick > MAX_TICK_AGE || nanos - sentNanos > MAX_NANOS) {
			if (++stale >= MAX_STALE) throw new IllegalStateException("Bridge inference repeatedly returned stale actions");
			return null;
		}
		stale = 0;
		return new Result<>(value, sentTick, epoch);
	}

	void cancel() {
		if (pending != null) pending.cancel(true);
		pending = null;
		stale = 0;
		epoch = UUID.randomUUID().toString();
	}
}