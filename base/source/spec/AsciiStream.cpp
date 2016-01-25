#include <ncs/spec/AsciiStream.h>

namespace ncs {

namespace spec {

AsciiStreamBase::AsciiStreamBase() {
}

AsciiStreamBase::~AsciiStreamBase() {
}

template<>
AsciiStream<sim::Bit>::AsciiStream(std::ostream& stream,
                                   DataSource* data_source)
  : stream_(stream),
    data_source_(data_source) {
  if (nullptr == data_source_) {
    return;
  }
  auto thread_function = [this]() {
    size_t num_elements = data_source_->getTotalNumberOfElements();
    size_t num_words = sim::Bit::num_words(num_elements);
    //auto& lambda_stream = &stream_;
    auto print_word = [this](sim::Bit::Word word) {
      for (size_t i = 0; i < sim::Bit::bits_per_word; ++i) {
        stream_ << sim::Bit::extract(word, i);
      }
    };
    while(true) {
      const void* data = data_source_->pull();
      if (nullptr == data) {
        break;
      }
      const sim::Bit::Word* d = static_cast<const sim::Bit::Word*>(data);
      print_word(d[0]);
      for (size_t i = 1; i < num_words; ++i) {
        print_word(d[i]);
      }
      stream_ << std::endl;
      data_source_->release();
    }
    delete data_source_;
    data_source_ = nullptr;
  };
  thread_ = std::thread(thread_function);
}

} // namespace spec

} // namespace ncs
