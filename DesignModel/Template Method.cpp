/*
����һ���������㷨�ĹǼܶ���һЩ�����ӳٵ������У�Tempplate Methodʹ����
���Բ��ı�һ���㷨�Ľṹ�����ض�����㷨��ĳЩ�ض����衣
*/
class Base {
public:
	Base() {};
	virtual ~Base() {};
	void f1() {};
	virtual void f2() = 0 {};
	void f3() {};
	virtual run() {
		f1();
		f2();
		f3();
	}
};

class EBase :public Base {
public:
	virtual void f2() {};
};

//  ���̶̹�����������һ����Ϊ�仯�������й��������ö�̬���ж�̬��
EBase().run()

