/*
����һϵ���㷨��������һ������װ��������ʹ���ǿ��໥�滻����ģʽʹ���㷨�ɶ�����ʹ��
���Ŀͻ�����

*/
class Base {
public:
	Base() {};
	virtual ~Base() {};
	virtual void f() = 0 {};
};

//ֱ��������϶�Base���к�����չ

class ExtendBase :public Base {
public:
	//...
private:
	Base * b;

};