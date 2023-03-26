#include <wx/wx.h>

int main(int argc, char** argv) {
  wxInitialize();
  wxInitAllImageHandlers();

  auto* app = new wxApp();
  auto* frame = new wxFrame(nullptr, wxID_ANY, "", wxDefaultPosition,
                            wxDefaultSize, wxSTAY_ON_TOP | wxFRAME_NO_TASKBAR);
  const auto imagePath =
      "/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/"
      "ToggleOff.png";
  auto* bitmap = new wxStaticBitmap(frame, wxID_ANY,
                                    wxBitmap(imagePath, wxBITMAP_TYPE_PNG));
  frame->SetSize(bitmap->GetSize());
  // The frame shows up without taking over the keyboard focus, which is quite
  // nice.
  frame->Show();
  app->MainLoop();
}
