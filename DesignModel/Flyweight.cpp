class Font {
private:
	string key;
public:
	Font(const string& key) {
		//...
	}
};

class FontFactory {
private:
	map<string, Font*> fontPool;

public:
	void clear() {
		fontPool.clear();
	}

	Font* getGont(const string& key) {
		if (fontPool.count(key) == 0) {
			Font* font = new Font(key);
			fontPool.insert({ key, font });
			return font;
		}
		else
			return fontPool[key];
	}
};