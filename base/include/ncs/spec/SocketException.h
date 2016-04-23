// SocketException class


#ifndef SocketException_class
#define SocketException_class

#include <string>

namespace ncs {

namespace spec {

class SocketException
{
 public:
  SocketException ( std::string s ) : m_s ( s ) {};
  ~SocketException (){};

  std::string description() { return m_s; }

 private:

  std::string m_s;

};

} // namespace spec

} // namespace ncs

#endif
