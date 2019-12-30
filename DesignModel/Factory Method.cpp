class Base {
public:
	Base();
	virtual void function() = 0;
	virtual ~Base();
};
class subBase :public Base {

public:
	subBase();
	~subBase();
	virtual void function() {}
};

class subFactory {
	Base* createFactory() {
		return new subBase();
	}
};
//利用工厂模式，将AF中的依赖具体类延迟，到工厂函数中
class A {
private:
	subFactory * sF;
public:
	A(subFactory* sF) :sF(sF) {}
	void AF() {

		Base *b = sF->createFactory();
	}
};