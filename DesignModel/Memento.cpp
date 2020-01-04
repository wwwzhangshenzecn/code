/*
在不破坏封装的前提下捕获一个对象的内部状态，并在该对象之外包存这个状态，
这样以后就可以将该对象恢复到保存点的状态
*/


class Memento {
private:
	string state:
	//...各种状态
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



