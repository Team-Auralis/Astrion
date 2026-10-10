@echo off
set "CUTECHESS=C:\Users\shaur\AppData\Local\Programs\Cute Chess\cutechess-cli.exe"

"%CUTECHESS%" -engine name="ORION" cmd="D:/orion-game-ai/orion_engine.bat" dir="D:/orion-game-ai" proto=uci -engine name="Stockfish" cmd="D:/orion-game-ai/stockfish.exe" proto=uci -each tc=40/60+1 -rounds 10 -pgnout results.pgn %*
