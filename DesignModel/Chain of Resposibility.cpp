/*
ʹ��������л��ᴦ�����󣬴Ӷ���������ķ����ߺͽ�����֮�����Ϲ�ϵ������д��������һ������
���������������ݸ�����֪����һ����������Ϊֹ��
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
				//...ȱʡ����
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