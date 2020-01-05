/*
使多个对象都有机会处理请求，从而避免请求的发送者和接收者之间的耦合关系，将这写对象连成一条链，
并沿着这条链传递该请求，知道有一个对象处理它为止。
*/



enum state{S,Q,M};

class Request {
private:
	state st;
public:
	Request(state s):st(s){}
	const state& getState() {
		return st;
	}
};


class HandlerRequest {
protected:
	virtual bool IsResponse() = 0;
	virtual TProcess() = 0;
	void GetProcess() {
		if (IsResponse())
			TProcess();
		else
			if next != nullptr:
				next->GetProcess();
			else
				//...缺省机制
			{}
	};
private:
	HandlerRequest* next;
	state st;
public:
	HandlerRequest(state st):st(st),next(nullptr){}
	void setNext(HandlerRequest* next) {
		this->next = next;
	}
};

class HanlderOne :public HandlerRequest {
public:
	HanlderOne(//...){//...}
	virtual bool IsResponse() {
		//....
	}
	virtual TProcess() {
		//...
	}
};