/*
通过对象创建模式绕开new，来避免对象创建过程中导致的紧耦合（对外依赖具体类）， 从而支持
对象创建的稳定。
*/
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