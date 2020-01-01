# 使用抽象工厂模式，创建一系列的关联对象，在工厂模式上有所改进

class IDBConnection {};
class IDBCommand {};
class IDBCReader {};
class Factory {
	virtual IDBConnection* createIDBConnection() = 0;
	virtual IDBCommand* createIDBCommand() = 0;
	virtual IDBCReader* createIDBReader() = 0;
};

class SQLDBConnection :public IDBConnection {};
class SQLDBCommand :public IDBCommand {};
class SQLReaf :public IDBCReader {};

class SQLFactory :public Factory {
	virtual IDBConnection* createIDBConnection() {
		return new IDBConnection();
	}
	virtual IDBCommand* createIDBCommand() {...}
	virtual IDBReader* createIDBReader() {...}
};

class ORACLEDBConnection :public IDBConnection {};
class ORACLEDBCommand :public IDBCommand {};
class ORACLEReaf :public IDBCReader {};

class ORACLEFactory :public Factory {
	virtual IDBConnection* createIDBConnection() {}
	virtual IDBCommand* createIDBCommand() {}
	virtual IDBCReader* createIDBReader() {}
};

class DB {
public:
	DB(Factory* f) :f(f) {}
	void DBC() {
		IDBConnection* conn = f->createIDBConnection();
		IDBCommand* comm = f->createIDBCommand();
		IDBCReader* read = f->createIDBReader();
		comm->connection(conn);
		read = comm->execute('sql', comm);
	}

private:
	Factory* f;
};