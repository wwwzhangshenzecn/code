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
//���ù���ģʽ����AF�е������������ӳ٣�������������
class A {
private:
	subFactory * sF;
public:
	A(subFactory* sF) :sF(sF) {}
	void AF() {

		Base *b = sF->createFactory();
	}
};