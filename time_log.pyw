from datetime import datetime
with open('SatrtLog.log', 'a', encoding='utf-8') as f:
    f.seek(0)
    data = datetime.now()
    f.write(str(data)+'\n')