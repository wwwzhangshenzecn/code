/*
�ڲ��ƻ���װ��ǰ���²���һ��������ڲ�״̬�����ڸö���֮��������״̬��
�����Ժ�Ϳ��Խ��ö���ָ���������״̬
*/


class Memento {
private:
	string state:
	//...����״̬
public	:
	Memento(Object* ob) :state(ob->state), ...{}
	void setState(Object* ob) {
		state = ob->state;
		//...
	}
	void getSatet(Object* ob) {
		s = ob->state;
		//...
	}
};

class Object {
private:
	string * state;
	//...
public:
	//....
	void getState(Memento* m) {
		state = m->getState();
		//...
	}
};

main() {
	Object* ob = new  Object();
	Memento* m = new Memento(ob);
	ob->getState(m);
}



