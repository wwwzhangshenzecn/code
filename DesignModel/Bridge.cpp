class ABase() {
public:
	ABase() {};
	virtual ~ABase() {};
	virtual void f() = 0 {};
}

class CABase :public ABase {
public:
	CABase();
	virtual void f() {}
	~CABase();
};

class DABase :public ABase {
public:
	DABase() {};
	virtual void f() {}
	~DABase() {};
};

class BBase {
public:
	BBase() {};
	virtual ~BBase() {};
	virtual void fB() = 0;
};
// 利用组合的方式将两个维度的类进行桥接起来
class DBBase :public BBase {
private:
	ABase *ab;
public:
	DBBase(ABase* ab):ab(ab) {};
	~DBBase() {};

	virtual fB() {
		ab->f();
	}
};
