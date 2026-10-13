package jwtc.android.chess.engine;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import jwtc.chess.board.BoardConstants;
import jwtc.android.chess.services.GameApi;
import jwtc.chess.JNI;
import jwtc.chess.Move;
import jwtc.chess.Pos;

/**
 * ORION AI Engine Adapter for Android Chess
 * Connects the Android app directly to the ORION Gaming Mind Neural Model & Search Engine.
 */
public class OrionEngine extends EngineApi {
    private static final String TAG = "OrionEngine";
    private static final Pattern MOVE_PATTERN = Pattern.compile("^([a-h][1-8])([a-h][1-8])([qrbnQRBN])?$");

    private final Context context;
    private final GameApi gameApi;
    private final JNI jni = JNI.getInstance();
    private String serverUrl = "http://10.0.2.2:8000"; // Default emulator localhost
    private Thread searchThread = null;
    private volatile boolean isAborted = false;

    public OrionEngine(Context context, GameApi gameApi, String serverUrl) {
        this.context = context;
        this.gameApi = gameApi;
        if (serverUrl != null && !serverUrl.isEmpty()) {
            this.serverUrl = serverUrl;
        }
    }

    public void setServerUrl(String url) {
        this.serverUrl = url;
    }

    public String getServerUrl() {
        return serverUrl;
    }

    @Override
    public void play() {
        Log.d(TAG, "ORION play requested. Ply: " + ply + ", Msecs: " + msecs);
        if (gameApi.isEnded()) {
            return;
        }

        for (EngineListener listener : listeners) {
            listener.OnEngineStarted();
        }

        isAborted = false;
        searchThread = new Thread(new RunnableOrionSearch());
        searchThread.start();
    }

    @Override
    public boolean isReady() {
        return searchThread == null || !searchThread.isAlive();
    }

    @Override
    public void abort(Runnable onDone) {
        isAborted = true;
        if (searchThread != null) {
            searchThread.interrupt();
            searchThread = null;
        }
        for (EngineListener listener : listeners) {
            listener.OnEngineAborted();
        }
        if (onDone != null) {
            updateHandler.post(onDone);
        }
    }

    @Override
    public void destroy() {
        abort(null);
    }

    private String[] getCandidateUrls() {
        if (serverUrl != null && !serverUrl.isEmpty() && !serverUrl.equals("http://127.0.0.1:8000") && !serverUrl.equals("http://10.0.2.2:8000")) {
            return new String[]{serverUrl, "http://127.0.0.1:8000", "http://10.0.2.2:8000"};
        }
        return new String[]{"http://127.0.0.1:8000", "http://10.0.2.2:8000"};
    }

    private class RunnableOrionSearch implements Runnable {
        @Override
        public void run() {
            try {
                if (gameApi.isEnded() || isAborted) {
                    return;
                }

                // 1. Get current position FEN
                String fen = jni.toFEN();
                int searchDepth = (ply > 0) ? ply : 2;

                // 2. Call Astrion API Server (try USB reverse proxy 127.0.0.1:8000, then emulator 10.0.2.2:8000)
                JSONObject res = null;
                for (String endpoint : getCandidateUrls()) {
                    if (isAborted) return;
                    try {
                        URL url = new URL(endpoint + "/api/move");
                        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("POST");
                        conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                        conn.setRequestProperty("Accept", "application/json");
                        conn.setConnectTimeout(1500);
                        conn.setReadTimeout(10000);
                        conn.setDoOutput(true);

                        JSONObject req = new JSONObject();
                        req.put("fen", fen);
                        req.put("depth", searchDepth);

                        byte[] postData = req.toString().getBytes(StandardCharsets.UTF_8);
                        try (OutputStream os = conn.getOutputStream()) {
                            os.write(postData);
                        }

                        int responseCode = conn.getResponseCode();
                        if (responseCode == 200) {
                            StringBuilder sb = new StringBuilder();
                            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                                String line;
                                while ((line = br.readLine()) != null) {
                                    sb.append(line);
                                }
                            }
                            res = new JSONObject(sb.toString());
                            serverUrl = endpoint;
                            break;
                        }
                    } catch (Exception ignored) {
                        // try next candidate endpoint
                    }
                }

                if (isAborted) return;

                if (res != null) {
                    String bestMoveUci = res.optString("best_move", "");
                    double score = res.optDouble("score", 0.0);
                    int nodes = res.optInt("nodes", 0);

                    Log.d(TAG, "Astrion AI chose: " + bestMoveUci + " (Eval: " + score + ", Nodes: " + nodes + ")");

                    int resolvedMove = resolveMove(bestMoveUci);
                    if (resolvedMove != 0) {
                        sendMoveMessageFromThread(resolvedMove, -1, (int) score);
                        sendMessageFromThread("Astrion AI (" + nodes + " nodes)\n\tEval: " + String.format("%.2f", score / 100.0), (float) (score / 100.0));
                        return;
                    }
                }

                Log.d(TAG, "Astrion server offline or unreachable, using local high-speed engine");
                fallbackLocalSearch((ply > 0) ? ply : 2);

            } catch (Exception ex) {
                Log.e(TAG, "Error running Astrion search: " + ex.getMessage() + ", falling back to local engine");
                fallbackLocalSearch((ply > 0) ? ply : 2);
            }
        }
    }

    private void fallbackLocalSearch(int searchDepth) {
        try {
            jni.searchDepth(searchDepth, quiescentSearchOn ? 1 : 0);
            int move = jni.getMove();
            int value = jni.peekSearchBestValue();
            sendMoveMessageFromThread(move, jni.getDuckMove(), value);
            sendMessageFromThread("ORION Offline Backup\n\t" + String.format("%.2f", (float) value / 100.0F), (float) value / 100.0F);
        } catch (Exception e) {
            sendErrorMessageFromThread();
        }
    }

    private int resolveMove(String uciMove) {
        Matcher matcher = MOVE_PATTERN.matcher(uciMove);
        if (!matcher.matches()) {
            return 0;
        }

        try {
            int from = Pos.fromString(matcher.group(1));
            int to = Pos.fromString(matcher.group(2));
            int promotionPiece = promotionPieceFromUci(matcher.group(3));

            int size = jni.getMoveArraySize();
            for (int i = 0; i < size; i++) {
                int move = jni.getMoveArrayAt(i);
                if (Move.getFrom(move) != from || Move.getTo(move) != to) {
                    continue;
                }
                if (promotionPiece != -1) {
                    if (!Move.isPromotionMove(move) || Move.getPromotionPiece(move) != promotionPiece) {
                        continue;
                    }
                }
                return move;
            }
        } catch (Exception ex) {
            Log.w(TAG, "Failed to resolve move " + uciMove, ex);
        }
        return 0;
    }

    private int promotionPieceFromUci(String promotion) {
        if (promotion == null || promotion.isEmpty()) {
            return -1;
        }
        switch (promotion.toLowerCase()) {
            case "q": return BoardConstants.QUEEN;
            case "r": return BoardConstants.ROOK;
            case "b": return BoardConstants.BISHOP;
            case "n": return BoardConstants.KNIGHT;
            default: return -1;
        }
    }
}
