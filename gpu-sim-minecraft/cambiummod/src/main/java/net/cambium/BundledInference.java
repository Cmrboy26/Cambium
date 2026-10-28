package net.cambium;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;

final class BundledInference {

	private static String resource() {
		String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
		String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
		String platform = os.contains("win") ? "windows" : os.contains("mac") ? "macos" : os.contains("linux") ? "linux" : "";
		String architecture = arch.equals("amd64") || arch.equals("x86_64") ? "x86_64"
				: arch.equals("aarch64") || arch.equals("arm64") ? "arm64" : "";
		if (platform.isEmpty() || architecture.isEmpty()) return null;
		return "/cambium-runtime/" + platform + "-" + architecture + "/cambium-inference" + (platform.equals("windows") ? ".exe" : "");
	}

	static boolean available() {
		String resource = resource();
		return resource != null && BundledInference.class.getResource(resource) != null
				&& BundledInference.class.getResource(resource + ".sha256") != null;
	}

	static Path prepare(Path game) throws IOException {
		String resource = resource();
		if (!available()) throw new IOException("No bundled inference runtime for this platform");
		String expected;
		try (InputStream input = BundledInference.class.getResourceAsStream(resource + ".sha256")) {
			expected = new String(input.readAllBytes(), java.nio.charset.StandardCharsets.US_ASCII).trim();
		}
		if (!expected.matches("[0-9a-f]{64}")) throw new IOException("Invalid bundled inference checksum");
		Path directory = game.resolve("sim-runtime").resolve(expected);
		Files.createDirectories(directory);
		Path target = directory.resolve(Path.of(resource).getFileName().toString());
		if (!Files.isRegularFile(target) || !digest(target).equals(expected)) {
			Path temporary = Files.createTempFile(directory, "inference-", ".tmp");
			try {
				try (InputStream input = BundledInference.class.getResourceAsStream(resource)) {
					Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING);
				}
				if (!digest(temporary).equals(expected)) throw new IOException("Bundled inference checksum mismatch");
				Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
			} finally {
				Files.deleteIfExists(temporary);
			}
		}
		if (!target.toFile().setExecutable(true) && !Files.isExecutable(target)) throw new IOException("Could not make inference runtime executable");
		return target;
	}

	private static String digest(Path path) throws IOException {
		try {
			MessageDigest digest = MessageDigest.getInstance("SHA-256");
			try (InputStream input = Files.newInputStream(path)) {
				byte[] buffer = new byte[1024 * 1024];
				int count;
				while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
			}
			return HexFormat.of().formatHex(digest.digest());
		} catch (NoSuchAlgorithmException error) {
			throw new IOException("SHA-256 is unavailable", error);
		}
	}

	static ProcessBuilder helper(Path executable, Path game, String action, String checkpoint) {
		String command = action.equals("local-world") ? "local" : action;
		var arguments = new ArrayList<>(List.of(executable.toString(), command, "--minecraft-directory", game.toString(), "--json", "--offline"));
		if (action.equals("import") || action.equals("local-policy")) arguments.addAll(List.of("--checkpoint", checkpoint));
		else arguments.addAll(List.of("--duel", game.resolve("sim-rosie-cache/duel.json").toString()));
		return new ProcessBuilder(arguments).directory(game.toFile())
				.redirectError(ProcessBuilder.Redirect.appendTo(game.resolve("logs/rosie-launcher.log").toFile()));
	}
}