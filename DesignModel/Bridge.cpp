/*
ʹ�á���������Ϲ�ϵ�������˳��ߺ�ʵ��֮����еİ󶨹�ϵ��ʹ�ó����ʵ�ֿ������Ÿ���
��ά�����仯��
*/
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
// ������ϵķ�ʽ������ά�ȵ�������Ž�����
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
