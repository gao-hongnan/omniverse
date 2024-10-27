from omnixamples.software_engineering.design_patterns.singleton.logger import Logger

logger1 = Logger()
logger2 = Logger()

assert logger1 is logger2
assert logger1 == logger2
print(f"logger1 id={id(logger1)} | logger2 id={id(logger2)}")
